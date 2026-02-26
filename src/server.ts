// @ts-nocheck
import { McpServer, ResourceTemplate } from "@modelcontextprotocol/sdk/server/mcp.js";
import { registerAppResource, registerAppTool, RESOURCE_MIME_TYPE } from "@modelcontextprotocol/ext-apps/server";
import { GoogleGenerativeAI } from "@google/generative-ai";
import { z } from "zod";
import * as fs from "fs/promises";
import * as path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// In-memory storage for generated images
const imageStore = new Map<string, { data: string; mimeType: string; metadata?: any }>();

// Auto-cleanup after 1 hour
function storeImage(id: string, data: string, mimeType: string, metadata?: any) {
  imageStore.set(id, { data, mimeType, metadata });
  setTimeout(() => imageStore.delete(id), 3600000); // 1 hour
}

function generateImageId(): string {
  return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
}

// Model selection enum
const ModelType = z.enum(["gemini-3.1-flash-image-preview", "gemini-3-pro-image-preview"]).default("gemini-3.1-flash-image-preview");

// Resolution options
// 0.5K (512px) is only supported by gemini-3.1-flash-image-preview
const ResolutionType = z.enum(["0.5K", "1K", "2K", "4K"]).default("1K");

// Aspect ratio options
const AspectRatioType = z.enum(["1:1", "1:4", "1:8", "2:3", "3:2", "3:4", "4:1", "4:3", "4:5", "5:4", "8:1", "9:16", "16:9", "21:9"]).optional();

// Common generation config schema
const GenerationConfigSchema = z.object({
  temperature: z.number().min(0).max(2).optional(),
  topP: z.number().min(0).max(1).optional(),
  topK: z.number().min(1).max(40).optional(),
  maxOutputTokens: z.number().optional(),
});

// Tool schemas for all tools
const GenerateImageArgsSchema = z.object({
  prompt: z.string().describe("The text prompt describing the image to generate"),
  outputDir: z.string().optional().describe("Directory to save the generated image (optional)"),
  model: ModelType.describe("Model to use"),
  resolution: ResolutionType.describe("Output resolution"),
  aspectRatio: AspectRatioType.describe("Aspect ratio"),
  returnBase64: z.boolean().optional().describe("Return as base64 (legacy mode)"),
  config: GenerationConfigSchema.optional().describe("Advanced generation configuration"),
});

const EditImageArgsSchema = z.object({
  prompt: z.string().describe("The text prompt describing how to edit the image"),
  imageData: z.string().optional().describe("Base64 encoded image data to edit"),
  imagePath: z.string().optional().describe("Path to the image file to edit"),
  outputDir: z.string().optional().describe("Directory to save the edited image"),
  model: ModelType.describe("Model to use"),
  resolution: ResolutionType.describe("Output resolution"),
  aspectRatio: AspectRatioType.describe("Aspect ratio"),
  returnBase64: z.boolean().optional().describe("Return as base64 (legacy mode)"),
  config: GenerationConfigSchema.optional().describe("Advanced generation configuration"),
}).refine(data => data.imageData || data.imagePath, {
  message: "Either imageData or imagePath must be provided",
});

const MultiImageEditArgsSchema = z.object({
  prompt: z.string().describe("The text prompt describing how to combine or edit the images"),
  images: z.array(z.object({
    imageData: z.string().optional().describe("Base64 encoded image data"),
    imagePath: z.string().optional().describe("Path to the image file"),
    description: z.string().optional().describe("Optional description of this image's role"),
  })).min(1).describe("Array of images to process"),
  outputDir: z.string().optional().describe("Directory to save the result"),
  model: ModelType.describe("Model to use"),
  resolution: ResolutionType.describe("Output resolution"),
  aspectRatio: AspectRatioType.describe("Aspect ratio"),
  returnBase64: z.boolean().optional().describe("Return as base64 (legacy mode)"),
  config: GenerationConfigSchema.optional().describe("Advanced generation configuration"),
}).refine(data => data.images.every(img => img.imageData || img.imagePath), {
  message: "Each image must have either imageData or imagePath",
});

const GenerateWithTemplateArgsSchema = z.object({
  template: z.enum([
    "photorealistic", "artistic", "logo", "portrait", "landscape",
    "product", "architectural", "fashion", "food", "abstract"
  ]).describe("Pre-defined prompt template"),
  customization: z.string().describe("Your specific requirements to customize the template"),
  outputDir: z.string().optional().describe("Directory to save the generated image"),
  model: ModelType.describe("Model to use"),
  resolution: ResolutionType.describe("Output resolution"),
  aspectRatio: AspectRatioType.describe("Aspect ratio"),
  returnBase64: z.boolean().optional().describe("Return as base64 (legacy mode)"),
  config: GenerationConfigSchema.optional().describe("Advanced generation configuration"),
});

const BatchGenerateArgsSchema = z.object({
  prompts: z.array(z.string()).min(1).describe("Array of prompts to generate images for"),
  outputDir: z.string().optional().describe("Directory to save the generated images"),
  model: ModelType.describe("Model to use"),
  resolution: ResolutionType.describe("Output resolution"),
  aspectRatio: AspectRatioType.describe("Aspect ratio"),
  config: GenerationConfigSchema.optional().describe("Advanced generation configuration"),
  parallel: z.boolean().optional().describe("Process prompts in parallel (default: false)"),
});

const GenerateVariationsArgsSchema = z.object({
  imagePath: z.string().optional().describe("Path to the reference image"),
  imageData: z.string().optional().describe("Base64 encoded reference image"),
  count: z.number().min(1).max(5).default(3).describe("Number of variations to generate (1-5)"),
  variationStrength: z.enum(["subtle", "moderate", "strong"]).default("moderate").describe("How different the variations should be"),
  outputDir: z.string().optional().describe("Directory to save the variations"),
  model: ModelType.describe("Model to use"),
  resolution: ResolutionType.describe("Output resolution"),
  aspectRatio: AspectRatioType.describe("Aspect ratio"),
  returnBase64: z.boolean().optional().describe("Return as base64 (legacy mode)"),
  config: GenerationConfigSchema.optional().describe("Advanced generation configuration"),
}).refine(data => data.imageData || data.imagePath, {
  message: "Either imageData or imagePath must be provided",
});

const AnalyzeImageArgsSchema = z.object({
  prompt: z.string().describe("Question or instruction about the image"),
  imageData: z.string().optional().describe("Base64 encoded image data to analyze"),
  imagePath: z.string().optional().describe("Path to the image file to analyze"),
  config: GenerationConfigSchema.optional().describe("Advanced generation configuration"),
}).refine(data => data.imageData || data.imagePath, {
  message: "Either imageData or imagePath must be provided",
});

const CompareImagesArgsSchema = z.object({
  image1Path: z.string().describe("Path to the first image"),
  image2Path: z.string().describe("Path to the second image"),
  compareType: z.enum(["differences", "similarities", "both"]).optional().describe("Type of comparison"),
});

// Prompt templates
const promptTemplates = {
  photorealistic: "Ultra-realistic photograph, professional photography, highly detailed, sharp focus, natural lighting, 8K resolution, shot with DSLR camera",
  artistic: "Artistic interpretation, creative style, expressive brushstrokes, vibrant colors, artistic composition, gallery-worthy artwork",
  logo: "Minimalist logo design, clean vector graphics, scalable, professional branding, modern design, simple geometric shapes, memorable icon",
  portrait: "Professional portrait photography, well-lit, shallow depth of field, bokeh background, natural skin tones, expressive eyes, studio lighting",
  landscape: "Breathtaking landscape photography, golden hour lighting, wide angle shot, dramatic sky, natural scenery, high dynamic range",
  product: "Product photography, white background, studio lighting, clean composition, commercial quality, detailed texture, professional presentation",
  architectural: "Architectural photography, precise lines, dramatic perspective, professional composition, detailed structure, impressive scale",
  fashion: "Fashion photography, editorial style, high-end fashion, professional model pose, stylish composition, magazine quality",
  food: "Food photography, appetizing presentation, professional styling, natural lighting, shallow depth of field, culinary art",
  abstract: "Abstract art, non-representational, creative composition, bold colors or monochrome, experimental style, artistic expression"
};

export function createServer(): McpServer {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error("GEMINI_API_KEY environment variable is required");
  }

  const genAI = new GoogleGenerativeAI(apiKey);

  const server = new McpServer({
    name: "nano-banana-mcp",
    version: "1.0.0",
  }, {
    capabilities: {
      tools: {},
      resources: {},
    },
  });

  const RESOURCE_URI = "ui://image-viewer/image-viewer.html";

  // Register image resources (base64 blob)
  server.registerResource(
    "generated-image",
    new ResourceTemplate("images://{id}", { list: undefined }),
    { description: "Generated image", mimeType: "image/png" },
    async (uri, vars) => {
      const id = Array.isArray(vars.id) ? vars.id[0] : vars.id;
      const imageData = imageStore.get(id);
      if (!imageData) {
        throw new Error(`Image not found: ${id}`);
      }

      return {
        contents: [{
          uri: uri.href,
          mimeType: imageData.mimeType,
          blob: imageData.data
        }]
      };
    }
  );

  // Helper function to build generation config
  function buildGenerationConfig(config: any, modelName: string, resolution?: string, aspectRatio?: string) {
    const generationConfig: any = config ? {
      temperature: config.temperature,
      topP: config.topP,
      topK: config.topK,
      maxOutputTokens: config.maxOutputTokens,
      responseModalities: ["TEXT", "IMAGE"],
    } : {
      responseModalities: ["TEXT", "IMAGE"],
    };

    if (resolution || aspectRatio) {
      generationConfig.imageConfig = {};
      if (resolution) {
        generationConfig.imageConfig.imageSize = resolution;
      }
      if (aspectRatio) {
        generationConfig.imageConfig.aspectRatio = aspectRatio;
      }
    }

    return generationConfig;
  }

  // Helper function to read image from path or data
  async function prepareImageData(imagePath?: string, imageData?: string): Promise<{ data: string; mimeType: string }> {
    if (imageData) {
      return { data: imageData, mimeType: "image/png" };
    } else if (imagePath) {
      const imageBuffer = await fs.readFile(imagePath);
      const data = imageBuffer.toString('base64');
      const ext = path.extname(imagePath).toLowerCase();
      const mimeType = ext === '.jpg' || ext === '.jpeg' ? 'image/jpeg' :
                       ext === '.gif' ? 'image/gif' :
                       ext === '.webp' ? 'image/webp' : 'image/png';
      return { data, mimeType };
    }
    throw new Error("Either imageData or imagePath must be provided");
  }

  // Register generate_image tool
  registerAppTool(
    server,
    "generate_image",
    {
      title: "Generate Image",
      description: "Generate an image using Gemini image models (Nano Banana 2 or Nano Banana Pro)",
      inputSchema: GenerateImageArgsSchema as any,
      _meta: { ui: { resourceUri: RESOURCE_URI } }
    },
    async (args: any) => {
      const { prompt, outputDir, model: modelName = "gemini-3.1-flash-image-preview", resolution, aspectRatio, returnBase64 = false, config } = args;

      const model = genAI.getGenerativeModel({ model: modelName });
      const generationConfig = buildGenerationConfig(config, modelName, resolution, aspectRatio);

      const result = await model.generateContent({
        contents: [{ role: "user", parts: [{ text: prompt }] }],
        generationConfig,
      });

      if (!result.response.candidates?.[0]?.content.parts) {
        throw new Error("No image was generated");
      }

      const imagePart = result.response.candidates[0].content.parts.find(part => 'inlineData' in part);
      if (!imagePart || !('inlineData' in imagePart)) {
        throw new Error("No image data found in the response");
      }

      const imageData = imagePart.inlineData!.data;
      const mimeType = imagePart.inlineData!.mimeType;

      // Legacy mode: return base64
      if (returnBase64) {
        return {
          content: [{ type: "image", data: imageData, mimeType }],
        };
      }

      // MCP Apps mode: store and return resource URI
      const imageId = generateImageId();
      const metadata = { prompt, resolution, aspectRatio, model: modelName, timestamp: new Date().toISOString() };
      storeImage(imageId, imageData, mimeType, metadata);

      // Also save to file if outputDir is specified
      if (outputDir) {
        const homeDir = process.env.HOME || process.env.USERPROFILE || ".";
        const defaultOutputDir = path.join(homeDir, "Downloads", "nano-banana-images");
        const finalOutputDir = outputDir || defaultOutputDir;
        const extension = mimeType === 'image/png' ? '.png' : mimeType === 'image/jpeg' ? '.jpg' : '.png';
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const filename = `generated-image-${timestamp}${extension}`;
        const filepath = path.join(finalOutputDir, filename);
        await fs.mkdir(finalOutputDir, { recursive: true });
        await fs.writeFile(filepath, Buffer.from(imageData, 'base64'));
      }

      return {
        content: [{ type: "text", text: JSON.stringify({ imageUri: `images://${imageId}`, metadata }) }],
        structuredContent: { imageUri: `images://${imageId}`, metadata }
      };
    }
  );

  // Register edit_image tool
  registerAppTool(
    server,
    "edit_image",
    {
      title: "Edit Image",
      description: "Edit an existing image using Gemini image models (Nano Banana 2 or Nano Banana Pro)",
      inputSchema: EditImageArgsSchema as any,
      _meta: { ui: { resourceUri: RESOURCE_URI } }
    },
    async (args: any) => {
      const { prompt, imageData, imagePath, outputDir, model: modelName = "gemini-3.1-flash-image-preview", resolution, aspectRatio, returnBase64 = false, config } = args;

      const model = genAI.getGenerativeModel({ model: modelName });
      const { data, mimeType } = await prepareImageData(imagePath, imageData);
      const imagePart = { inlineData: { data, mimeType } };

      const generationConfig = buildGenerationConfig(config, modelName, resolution, aspectRatio);

      const result = await model.generateContent({
        contents: [{ role: "user", parts: [{ text: prompt }, imagePart] }],
        generationConfig,
      });

      if (!result.response.candidates?.[0]?.content.parts) {
        throw new Error("No edited image was generated");
      }

      const outputImagePart = result.response.candidates[0].content.parts.find(part => 'inlineData' in part);
      if (!outputImagePart || !('inlineData' in outputImagePart)) {
        throw new Error("No image data found in the response");
      }

      const outputImageData = outputImagePart.inlineData!.data;
      const outputMimeType = outputImagePart.inlineData!.mimeType;

      // Legacy mode
      if (returnBase64) {
        return {
          content: [{ type: "image", data: outputImageData, mimeType: outputMimeType }],
        };
      }

      // MCP Apps mode
      const imageId = generateImageId();
      const metadata = { prompt, resolution, aspectRatio, model: modelName, timestamp: new Date().toISOString(), operation: "edit" };
      storeImage(imageId, outputImageData, outputMimeType, metadata);

      // Save to file if outputDir specified
      if (outputDir) {
        const homeDir = process.env.HOME || process.env.USERPROFILE || ".";
        const defaultOutputDir = path.join(homeDir, "Downloads", "nano-banana-images");
        const finalOutputDir = outputDir || defaultOutputDir;
        const extension = outputMimeType === 'image/png' ? '.png' : outputMimeType === 'image/jpeg' ? '.jpg' : '.png';
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const filename = `edited-image-${timestamp}${extension}`;
        const filepath = path.join(finalOutputDir, filename);
        await fs.mkdir(finalOutputDir, { recursive: true });
        await fs.writeFile(filepath, Buffer.from(outputImageData, 'base64'));
      }

      return {
        content: [{ type: "text", text: JSON.stringify({ imageUri: `images://${imageId}`, metadata }) }],
        structuredContent: { imageUri: `images://${imageId}`, metadata }
      };
    }
  );

  // Register multi_image_edit tool
  registerAppTool(
    server,
    "multi_image_edit",
    {
      title: "Multi-Image Edit",
      description: "Edit or combine multiple images using Gemini image models (Nano Banana 2 or Nano Banana Pro)",
      inputSchema: MultiImageEditArgsSchema as any,
      _meta: { ui: { resourceUri: RESOURCE_URI } }
    },
    async (args: any) => {
      const { prompt, images, outputDir, model: modelName = "gemini-3.1-flash-image-preview", resolution, aspectRatio, returnBase64 = false, config } = args;

      const model = genAI.getGenerativeModel({ model: modelName });

      // Process all images
      const imageParts = [];
      for (const image of images) {
        const { data, mimeType } = await prepareImageData(image.imagePath, image.imageData);
        imageParts.push({ inlineData: { data, mimeType } });
      }

      const generationConfig = buildGenerationConfig(config, modelName, resolution, aspectRatio);

      const result = await model.generateContent({
        contents: [{ role: "user", parts: [{ text: prompt }, ...imageParts] }],
        generationConfig,
      });

      if (!result.response.candidates?.[0]?.content.parts) {
        throw new Error("No multi-image result was generated");
      }

      const outputImagePart = result.response.candidates[0].content.parts.find(part => 'inlineData' in part);
      if (!outputImagePart || !('inlineData' in outputImagePart)) {
        throw new Error("No image data found in the response");
      }

      const outputImageData = outputImagePart.inlineData!.data;
      const outputMimeType = outputImagePart.inlineData!.mimeType;

      // Legacy mode
      if (returnBase64) {
        return {
          content: [{ type: "image", data: outputImageData, mimeType: outputMimeType }],
        };
      }

      // MCP Apps mode
      const imageId = generateImageId();
      const metadata = { prompt, resolution, aspectRatio, model: modelName, timestamp: new Date().toISOString(), operation: "multi-edit" };
      storeImage(imageId, outputImageData, outputMimeType, metadata);

      return {
        content: [{ type: "text", text: JSON.stringify({ imageUri: `images://${imageId}`, metadata }) }],
        structuredContent: { imageUri: `images://${imageId}`, metadata }
      };
    }
  );

  // Register generate_with_template tool
  registerAppTool(
    server,
    "generate_with_template",
    {
      title: "Generate with Template",
      description: "Generate an image using a pre-defined style template",
      inputSchema: GenerateWithTemplateArgsSchema as any,
      _meta: { ui: { resourceUri: RESOURCE_URI } }
    },
    async (args: any) => {
      const { template, customization, outputDir, model: modelName = "gemini-3.1-flash-image-preview", resolution, aspectRatio, returnBase64 = false, config } = args;

      const model = genAI.getGenerativeModel({ model: modelName });
      const templatePrompt = promptTemplates[template as keyof typeof promptTemplates];
      const fullPrompt = `${templatePrompt}. ${customization}`;

      const generationConfig = buildGenerationConfig(config, modelName, resolution, aspectRatio);

      const result = await model.generateContent({
        contents: [{ role: "user", parts: [{ text: fullPrompt }] }],
        generationConfig,
      });

      if (!result.response.candidates?.[0]?.content.parts) {
        throw new Error("No image was generated");
      }

      const imagePart = result.response.candidates[0].content.parts.find(part => 'inlineData' in part);
      if (!imagePart || !('inlineData' in imagePart)) {
        throw new Error("No image data found in the response");
      }

      const imageData = imagePart.inlineData!.data;
      const mimeType = imagePart.inlineData!.mimeType;

      // Legacy mode
      if (returnBase64) {
        return {
          content: [{ type: "image", data: imageData, mimeType }],
        };
      }

      // MCP Apps mode
      const imageId = generateImageId();
      const metadata = { prompt: fullPrompt, template, resolution, aspectRatio, model: modelName, timestamp: new Date().toISOString() };
      storeImage(imageId, imageData, mimeType, metadata);

      return {
        content: [{ type: "text", text: JSON.stringify({ imageUri: `images://${imageId}`, metadata }) }],
        structuredContent: { imageUri: `images://${imageId}`, metadata }
      };
    }
  );

  // Register batch_generate tool
  registerAppTool(
    server,
    "batch_generate",
    {
      title: "Batch Generate",
      description: "Generate multiple images from an array of prompts",
      inputSchema: BatchGenerateArgsSchema as any,
      _meta: { ui: { resourceUri: RESOURCE_URI } }
    },
    async (args: any) => {
      const { prompts, outputDir, model: modelName = "gemini-3.1-flash-image-preview", resolution, aspectRatio, config, parallel = false } = args;

      const model = genAI.getGenerativeModel({ model: modelName });
      const generationConfig = buildGenerationConfig(config, modelName, resolution, aspectRatio);

      const images: any[] = [];

      if (parallel) {
        const promises = prompts.map(async (prompt: string, index: number) => {
          try {
            const result = await model.generateContent({
              contents: [{ role: "user", parts: [{ text: prompt }] }],
              generationConfig,
            });

            if (result.response.candidates?.[0]?.content.parts) {
              const imagePart = result.response.candidates[0].content.parts.find(part => 'inlineData' in part);
              if (imagePart && 'inlineData' in imagePart) {
                const imageData = imagePart.inlineData!.data;
                const mimeType = imagePart.inlineData!.mimeType;
                const imageId = generateImageId();
                const metadata = { prompt, resolution, aspectRatio, model: modelName, timestamp: new Date().toISOString(), index: index + 1 };
                storeImage(imageId, imageData, mimeType, metadata);
                return { imageUri: `images://${imageId}`, metadata };
              }
            }
            return null;
          } catch (error) {
            console.error(`Failed to generate image ${index + 1}:`, error);
            return null;
          }
        });

        const results = await Promise.all(promises);
        images.push(...results.filter(r => r !== null));
      } else {
        for (let i = 0; i < prompts.length; i++) {
          try {
            const result = await model.generateContent({
              contents: [{ role: "user", parts: [{ text: prompts[i] }] }],
              generationConfig,
            });

            if (result.response.candidates?.[0]?.content.parts) {
              const imagePart = result.response.candidates[0].content.parts.find(part => 'inlineData' in part);
              if (imagePart && 'inlineData' in imagePart) {
                const imageData = imagePart.inlineData!.data;
                const mimeType = imagePart.inlineData!.mimeType;
                const imageId = generateImageId();
                const metadata = { prompt: prompts[i], resolution, aspectRatio, model: modelName, timestamp: new Date().toISOString(), index: i + 1 };
                storeImage(imageId, imageData, mimeType, metadata);
                images.push({ imageUri: `images://${imageId}`, metadata });
              }
            }
          } catch (error) {
            console.error(`Failed to generate image ${i + 1}:`, error);
          }
        }
      }

      return {
        content: [{ type: "text", text: JSON.stringify({ images }) }],
        structuredContent: { images }
      };
    }
  );

  // Register generate_variations tool
  registerAppTool(
    server,
    "generate_variations",
    {
      title: "Generate Variations",
      description: "Generate variations of an existing image",
      inputSchema: GenerateVariationsArgsSchema as any,
      _meta: { ui: { resourceUri: RESOURCE_URI } }
    },
    async (args: any) => {
      const { imagePath, imageData, count = 3, variationStrength = "moderate", outputDir, model: modelName = "gemini-3.1-flash-image-preview", resolution, aspectRatio, returnBase64 = false, config } = args;

      const model = genAI.getGenerativeModel({ model: modelName });
      const { data, mimeType } = await prepareImageData(imagePath, imageData);
      const imagePart = { inlineData: { data, mimeType } };

      const variationPrompts: Record<string, string> = {
        subtle: "Create a very similar variation of this image with minimal changes, keeping the same style and composition",
        moderate: "Create a variation of this image with moderate changes while maintaining the core concept and style",
        strong: "Create a significantly different variation of this image, exploring new interpretations while keeping the main subject"
      };

      const basePrompt = variationPrompts[variationStrength];
      const temperature = variationStrength === "subtle" ? 0.3 : variationStrength === "moderate" ? 0.7 : 1.2;

      const generationConfig: any = config ? {
        ...config,
        temperature: config.temperature ?? temperature,
        responseModalities: ["TEXT", "IMAGE"],
      } : {
        temperature,
        responseModalities: ["TEXT", "IMAGE"],
      };

      if (resolution || aspectRatio) {
        generationConfig.imageConfig = {};
        if (resolution) {
          generationConfig.imageConfig.imageSize = resolution;
        }
        if (aspectRatio) {
          generationConfig.imageConfig.aspectRatio = aspectRatio;
        }
      }

      const images: any[] = [];

      for (let i = 0; i < count; i++) {
        try {
          const result = await model.generateContent({
            contents: [{ role: "user", parts: [{ text: `${basePrompt} (variation ${i + 1} of ${count})` }, imagePart] }],
            generationConfig,
          });

          if (result.response.candidates?.[0]?.content.parts) {
            const outputImagePart = result.response.candidates[0].content.parts.find(part => 'inlineData' in part);
            if (outputImagePart && 'inlineData' in outputImagePart) {
              const outputImageData = outputImagePart.inlineData!.data;
              const outputMimeType = outputImagePart.inlineData!.mimeType;
              const imageId = generateImageId();
              const metadata = {
                variationOf: imagePath || "provided image",
                variationStrength,
                resolution,
                aspectRatio,
                model: modelName,
                timestamp: new Date().toISOString(),
                index: i + 1
              };
              storeImage(imageId, outputImageData, outputMimeType, metadata);
              images.push({ imageUri: `images://${imageId}`, metadata });
            }
          }
        } catch (error) {
          console.error(`Failed to generate variation ${i + 1}:`, error);
        }
      }

      if (returnBase64) {
        // Legacy mode: return first variation as base64
        if (images.length > 0) {
          const firstImage = imageStore.get(images[0].imageUri.replace("images://", ""));
          if (firstImage) {
            return {
              content: [{ type: "image", data: firstImage.data, mimeType: firstImage.mimeType }],
            };
          }
        }
      }

      return {
        content: [{ type: "text", text: JSON.stringify({ images }) }],
        structuredContent: { images }
      };
    }
  );

  // Register analyze_image tool (no MCP Apps UI, just returns text)
  server.registerTool(
    "analyze_image",
    {
      title: "Analyze Image",
      description: "Analyze an image and answer questions about it using Gemini",
      inputSchema: AnalyzeImageArgsSchema,
    },
    async (args: any) => {
      const { prompt, imageData, imagePath, config } = args;

      const model = genAI.getGenerativeModel({ model: "gemini-2.5-flash" });
      const { data, mimeType } = await prepareImageData(imagePath, imageData);
      const imagePart = { inlineData: { data, mimeType } };

      const generationConfig = config ? {
        temperature: config.temperature,
        topP: config.topP,
        topK: config.topK,
        maxOutputTokens: config.maxOutputTokens,
      } : {};

      const result = await model.generateContent({
        contents: [{ role: "user", parts: [{ text: prompt }, imagePart] }],
        generationConfig,
      });

      if (!result.response.text) {
        throw new Error("No response text generated");
      }

      return {
        content: [{ type: "text", text: result.response.text() }],
      };
    }
  );

  // Register compare_images tool (no MCP Apps UI, just returns text)
  server.registerTool(
    "compare_images",
    {
      title: "Compare Images",
      description: "Compare two images and analyze their differences or similarities",
      inputSchema: CompareImagesArgsSchema,
    },
    async (args: any) => {
      const { image1Path, image2Path, compareType = "both" } = args;

      const model = genAI.getGenerativeModel({ model: "gemini-2.5-flash" });

      const image1 = await prepareImageData(image1Path);
      const image2 = await prepareImageData(image2Path);

      const image1Part = { inlineData: { data: image1.data, mimeType: image1.mimeType } };
      const image2Part = { inlineData: { data: image2.data, mimeType: image2.mimeType } };

      let prompt = "";
      switch (compareType) {
        case "differences":
          prompt = "Compare these two images and describe all the differences between them in detail.";
          break;
        case "similarities":
          prompt = "Compare these two images and describe all the similarities between them in detail.";
          break;
        case "both":
          prompt = "Compare these two images. First, describe their similarities, then describe their differences. Be thorough and detailed.";
          break;
      }

      const result = await model.generateContent({
        contents: [{ role: "user", parts: [{ text: prompt }, image1Part, image2Part] }],
      });

      if (!result.response.text) {
        throw new Error("No comparison analysis generated");
      }

      return {
        content: [{ type: "text", text: `Image Comparison (${compareType}):\n\n${result.response.text()}` }],
      };
    }
  );

  // Register UI resource
  registerAppResource(
    server,
    RESOURCE_URI,
    RESOURCE_URI,
    { mimeType: RESOURCE_MIME_TYPE },
    async () => {
      const htmlPath = path.join(__dirname, "image-viewer.html");
      const html = await fs.readFile(htmlPath, "utf-8");
      return {
        contents: [{ uri: RESOURCE_URI, mimeType: RESOURCE_MIME_TYPE, text: html }]
      };
    }
  );

  return server;
}
