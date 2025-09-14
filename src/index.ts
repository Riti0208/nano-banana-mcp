#!/usr/bin/env node

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  ToolSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { GoogleGenerativeAI } from "@google/generative-ai";
import { z } from "zod";
import * as fs from "fs/promises";
import * as path from "path";

// Common generation config schema
const GenerationConfigSchema = z.object({
  temperature: z.number().min(0).max(2).optional().describe("Controls randomness (0.0-2.0, default: 1.0)"),
  topP: z.number().min(0).max(1).optional().describe("Nucleus sampling threshold (0.0-1.0)"),
  topK: z.number().min(1).max(40).optional().describe("Top-k sampling (1-40)"),
  maxOutputTokens: z.number().optional().describe("Maximum number of output tokens"),
});

const GenerateImageArgsSchema = z.object({
  prompt: z.string().describe("The text prompt describing the image to generate"),
  outputDir: z.string().optional().describe("Directory to save the generated image (optional)"),
  config: GenerationConfigSchema.optional().describe("Advanced generation configuration"),
});

const EditImageArgsSchema = z.object({
  prompt: z.string().describe("The text prompt describing how to edit the image"),
  imageData: z.string().optional().describe("Base64 encoded image data to edit"),
  imagePath: z.string().optional().describe("Path to the image file to edit"),
  outputDir: z.string().optional().describe("Directory to save the edited image (optional)"),
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

const MultiImageEditArgsSchema = z.object({
  prompt: z.string().describe("The text prompt describing how to combine or edit multiple images"),
  images: z.array(z.object({
    imageData: z.string().optional().describe("Base64 encoded image data"),
    imagePath: z.string().optional().describe("Path to the image file"),
    description: z.string().optional().describe("Optional description of this image's role"),
  })).min(1).describe("Array of images to process"),
  outputDir: z.string().optional().describe("Directory to save the result (optional)"),
  config: GenerationConfigSchema.optional().describe("Advanced generation configuration"),
}).refine(data => data.images.every(img => img.imageData || img.imagePath), {
  message: "Each image must have either imageData or imagePath",
});

// Batch processing schema
const BatchGenerateArgsSchema = z.object({
  prompts: z.array(z.string()).min(1).describe("Array of prompts to generate images for"),
  outputDir: z.string().optional().describe("Directory to save the generated images"),
  config: GenerationConfigSchema.optional().describe("Advanced generation configuration"),
  parallel: z.boolean().optional().describe("Process prompts in parallel (default: false)"),
});

// Compare images schema
const CompareImagesArgsSchema = z.object({
  image1Path: z.string().describe("Path to the first image"),
  image2Path: z.string().describe("Path to the second image"),
  compareType: z.enum(["differences", "similarities", "both"]).optional().describe("Type of comparison"),
});

// Image variations schema
const GenerateVariationsArgsSchema = z.object({
  imagePath: z.string().optional().describe("Path to the reference image"),
  imageData: z.string().optional().describe("Base64 encoded reference image"),
  count: z.number().min(1).max(5).default(3).describe("Number of variations to generate (1-5)"),
  variationStrength: z.enum(["subtle", "moderate", "strong"]).default("moderate").describe("How different the variations should be"),
  outputDir: z.string().optional().describe("Directory to save the variations"),
  config: GenerationConfigSchema.optional().describe("Advanced generation configuration"),
}).refine(data => data.imageData || data.imagePath, {
  message: "Either imageData or imagePath must be provided",
});

// Prompt templates
const PromptTemplateArgsSchema = z.object({
  template: z.enum([
    "photorealistic",
    "artistic",
    "logo",
    "portrait",
    "landscape",
    "product",
    "architectural",
    "fashion",
    "food",
    "abstract"
  ]).describe("Pre-defined prompt template"),
  customization: z.string().describe("Your specific requirements to customize the template"),
  outputDir: z.string().optional().describe("Directory to save the generated image"),
  config: GenerationConfigSchema.optional().describe("Advanced generation configuration"),
});

class NanaBananaMCPServer {
  private server: Server;
  private genAI: GoogleGenerativeAI;
  
  // Prompt templates mapping
  private promptTemplates = {
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

  constructor() {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
      throw new Error("GEMINI_API_KEY environment variable is required");
    }

    this.genAI = new GoogleGenerativeAI(apiKey);
    this.server = new Server(
      {
        name: "nano-banana-mcp",
        version: "1.0.0",
      },
      {
        capabilities: {
          tools: {},
        },
      }
    );

    this.setupToolHandlers();
  }

  private setupToolHandlers() {
    this.server.setRequestHandler(ListToolsRequestSchema, async () => {
      return {
        tools: [
          {
            name: "generate_image",
            description: "Generate an image using Gemini 2.5 Flash Image Preview (nano-banana)",
            inputSchema: {
              type: "object",
              properties: {
                prompt: {
                  type: "string",
                  description: "The text prompt describing the image to generate",
                },
                outputDir: {
                  type: "string",
                  description: "Directory to save the generated image (optional, defaults to current directory)",
                },
              },
              required: ["prompt"],
            },
          },
          {
            name: "edit_image",
            description: "Edit an existing image using Gemini 2.5 Flash Image Preview",
            inputSchema: {
              type: "object",
              properties: {
                prompt: {
                  type: "string",
                  description: "The text prompt describing how to edit the image",
                },
                imageData: {
                  type: "string",
                  description: "Base64 encoded image data to edit (optional if imagePath is provided)",
                },
                imagePath: {
                  type: "string",
                  description: "Path to the image file to edit (optional if imageData is provided)",
                },
                outputDir: {
                  type: "string",
                  description: "Directory to save the edited image (optional, defaults to current directory)",
                },
              },
              required: ["prompt"],
            },
          },
          {
            name: "analyze_image",
            description: "Analyze an image and answer questions about it using Gemini",
            inputSchema: {
              type: "object",
              properties: {
                prompt: {
                  type: "string",
                  description: "Question or instruction about the image",
                },
                imageData: {
                  type: "string",
                  description: "Base64 encoded image data to analyze (optional if imagePath is provided)",
                },
                imagePath: {
                  type: "string",
                  description: "Path to the image file to analyze (optional if imageData is provided)",
                },
              },
              required: ["prompt"],
            },
          },
          {
            name: "multi_image_edit",
            description: "Edit or combine multiple images using Gemini 2.5 Flash Image Preview (e.g., transfer pose, style, combine elements)",
            inputSchema: {
              type: "object",
              properties: {
                prompt: {
                  type: "string",
                  description: "The text prompt describing how to combine or edit the images",
                },
                images: {
                  type: "array",
                  description: "Array of images to process",
                  items: {
                    type: "object",
                    properties: {
                      imageData: {
                        type: "string",
                        description: "Base64 encoded image data (optional if imagePath is provided)",
                      },
                      imagePath: {
                        type: "string",
                        description: "Path to the image file (optional if imageData is provided)",
                      },
                      description: {
                        type: "string",
                        description: "Optional description of this image's role (e.g., 'reference pose', 'target person')",
                      },
                    },
                  },
                  minItems: 1,
                },
                outputDir: {
                  type: "string",
                  description: "Directory to save the result (optional, defaults to current directory)",
                },
              },
              required: ["prompt", "images"],
            },
          },
          {
            name: "batch_generate",
            description: "Generate multiple images from an array of prompts",
            inputSchema: {
              type: "object",
              properties: {
                prompts: {
                  type: "array",
                  description: "Array of prompts to generate images for",
                  items: { type: "string" },
                  minItems: 1,
                },
                outputDir: {
                  type: "string",
                  description: "Directory to save the generated images",
                },
                config: {
                  type: "object",
                  description: "Advanced generation configuration",
                },
                parallel: {
                  type: "boolean",
                  description: "Process prompts in parallel (default: false)",
                },
              },
              required: ["prompts"],
            },
          },
          {
            name: "generate_variations",
            description: "Generate variations of an existing image",
            inputSchema: {
              type: "object",
              properties: {
                imagePath: {
                  type: "string",
                  description: "Path to the reference image",
                },
                imageData: {
                  type: "string",
                  description: "Base64 encoded reference image",
                },
                count: {
                  type: "number",
                  description: "Number of variations to generate (1-5)",
                  minimum: 1,
                  maximum: 5,
                  default: 3,
                },
                variationStrength: {
                  type: "string",
                  enum: ["subtle", "moderate", "strong"],
                  description: "How different the variations should be",
                  default: "moderate",
                },
                outputDir: {
                  type: "string",
                  description: "Directory to save the variations",
                },
                config: {
                  type: "object",
                  description: "Advanced generation configuration",
                },
              },
              required: [],
            },
          },
          {
            name: "generate_with_template",
            description: "Generate an image using a pre-defined style template",
            inputSchema: {
              type: "object",
              properties: {
                template: {
                  type: "string",
                  enum: ["photorealistic", "artistic", "logo", "portrait", "landscape", "product", "architectural", "fashion", "food", "abstract"],
                  description: "Pre-defined prompt template",
                },
                customization: {
                  type: "string",
                  description: "Your specific requirements to customize the template",
                },
                outputDir: {
                  type: "string",
                  description: "Directory to save the generated image",
                },
                config: {
                  type: "object",
                  description: "Advanced generation configuration",
                },
              },
              required: ["template", "customization"],
            },
          },
          {
            name: "compare_images",
            description: "Compare two images and analyze their differences or similarities",
            inputSchema: {
              type: "object",
              properties: {
                image1Path: {
                  type: "string",
                  description: "Path to the first image",
                },
                image2Path: {
                  type: "string",
                  description: "Path to the second image",
                },
                compareType: {
                  type: "string",
                  enum: ["differences", "similarities", "both"],
                  description: "Type of comparison",
                  default: "both",
                },
              },
              required: ["image1Path", "image2Path"],
            },
          },
        ],
      };
    });

    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      const { name, arguments: args } = request.params;

      try {
        switch (name) {
          case "generate_image":
            return await this.handleGenerateImage(args);
          case "edit_image":
            return await this.handleEditImage(args);
          case "analyze_image":
            return await this.handleAnalyzeImage(args);
          case "multi_image_edit":
            return await this.handleMultiImageEdit(args);
          case "batch_generate":
            return await this.handleBatchGenerate(args);
          case "generate_variations":
            return await this.handleGenerateVariations(args);
          case "generate_with_template":
            return await this.handleGenerateWithTemplate(args);
          case "compare_images":
            return await this.handleCompareImages(args);
          default:
            throw new Error(`Unknown tool: ${name}`);
        }
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : String(error);
        return {
          content: [
            {
              type: "text",
              text: `Error: ${errorMessage}`,
            },
          ],
          isError: true,
        };
      }
    });
  }

  private async handleGenerateImage(args: unknown) {
    const { prompt, outputDir = ".", config } = GenerateImageArgsSchema.parse(args);

    const model = this.genAI.getGenerativeModel({ 
      model: "gemini-2.5-flash-image-preview" 
    });

    // Build generation config
    const generationConfig = config ? {
      temperature: config.temperature,
      topP: config.topP,
      topK: config.topK,
      maxOutputTokens: config.maxOutputTokens,
      responseModalities: ["TEXT", "IMAGE"],
    } : {
      responseModalities: ["TEXT", "IMAGE"],
    };

    const result = await model.generateContent({
      contents: [{ role: "user", parts: [{ text: prompt }] }],
      generationConfig,
    });
    
    if (!result.response.candidates || result.response.candidates.length === 0) {
      throw new Error("No image was generated");
    }

    const candidate = result.response.candidates[0];
    if (!candidate.content.parts || candidate.content.parts.length === 0) {
      throw new Error("No content parts in the response");
    }

    // Find the image part in the response
    const imagePart = candidate.content.parts.find(part => 'inlineData' in part);
    
    if (!imagePart || !('inlineData' in imagePart)) {
      throw new Error("No image data found in the response");
    }

    const imageData = imagePart.inlineData!.data;
    const mimeType = imagePart.inlineData!.mimeType;
    
    // Determine file extension from MIME type
    const extension = mimeType === 'image/png' ? '.png' : 
                     mimeType === 'image/jpeg' ? '.jpg' : 
                     '.png'; // default to PNG

    // Create filename with timestamp
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const filename = `generated-image-${timestamp}${extension}`;
    const filepath = path.join(outputDir, filename);

    // Save the image
    await fs.mkdir(outputDir, { recursive: true });
    await fs.writeFile(filepath, Buffer.from(imageData, 'base64'));

    return {
      content: [
        {
          type: "text",
          text: `Image generated successfully and saved to: ${filepath}`,
        },
      ],
    };
  }

  private async handleEditImage(args: unknown) {
    const { prompt, imageData, imagePath, outputDir = ".", config } = EditImageArgsSchema.parse(args);

    const model = this.genAI.getGenerativeModel({ 
      model: "gemini-2.5-flash-image-preview" 
    });

    let finalImageData: string;
    let mimeType: string;

    if (imageData) {
      finalImageData = imageData;
      mimeType = "image/png"; // Default assumption
    } else if (imagePath) {
      // Read the image file and convert to base64
      const imageBuffer = await fs.readFile(imagePath);
      finalImageData = imageBuffer.toString('base64');
      
      // Determine MIME type from file extension
      const ext = path.extname(imagePath).toLowerCase();
      switch (ext) {
        case '.png':
          mimeType = 'image/png';
          break;
        case '.jpg':
        case '.jpeg':
          mimeType = 'image/jpeg';
          break;
        case '.gif':
          mimeType = 'image/gif';
          break;
        case '.webp':
          mimeType = 'image/webp';
          break;
        default:
          mimeType = 'image/png'; // Default fallback
      }
    } else {
      throw new Error("Either imageData or imagePath must be provided");
    }

    // Create the image part for input
    const imagePart = {
      inlineData: {
        data: finalImageData,
        mimeType: mimeType,
      },
    };

    // Build generation config
    const generationConfig = config ? {
      temperature: config.temperature,
      topP: config.topP,
      topK: config.topK,
      maxOutputTokens: config.maxOutputTokens,
      responseModalities: ["TEXT", "IMAGE"],
    } : {
      responseModalities: ["TEXT", "IMAGE"],
    };

    const result = await model.generateContent({
      contents: [{ role: "user", parts: [{ text: prompt }, imagePart] }],
      generationConfig,
    });
    
    if (!result.response.candidates || result.response.candidates.length === 0) {
      throw new Error("No edited image was generated");
    }

    const candidate = result.response.candidates[0];
    if (!candidate.content.parts || candidate.content.parts.length === 0) {
      throw new Error("No content parts in the response");
    }

    // Find the image part in the response
    const outputImagePart = candidate.content.parts.find(part => 'inlineData' in part);
    
    if (!outputImagePart || !('inlineData' in outputImagePart)) {
      throw new Error("No image data found in the response");
    }

    const outputImageData = outputImagePart.inlineData!.data;
    const outputMimeType = outputImagePart.inlineData!.mimeType;
    
    // Determine file extension from MIME type
    const extension = outputMimeType === 'image/png' ? '.png' : 
                     outputMimeType === 'image/jpeg' ? '.jpg' : 
                     '.png'; // default to PNG

    // Create filename with timestamp
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const filename = `edited-image-${timestamp}${extension}`;
    const filepath = path.join(outputDir, filename);

    // Save the edited image
    await fs.mkdir(outputDir, { recursive: true });
    await fs.writeFile(filepath, Buffer.from(outputImageData, 'base64'));

    return {
      content: [
        {
          type: "text",
          text: `Image edited successfully and saved to: ${filepath}`,
        },
      ],
    };
  }

  private async handleAnalyzeImage(args: unknown) {
    const { prompt, imageData, imagePath, config } = AnalyzeImageArgsSchema.parse(args);

    const model = this.genAI.getGenerativeModel({ 
      model: "gemini-2.5-flash" // Using regular Flash for analysis, not image generation
    });

    let finalImageData: string;
    let mimeType: string;

    if (imageData) {
      finalImageData = imageData;
      mimeType = "image/png"; // Default assumption
    } else if (imagePath) {
      // Read the image file and convert to base64
      const imageBuffer = await fs.readFile(imagePath);
      finalImageData = imageBuffer.toString('base64');
      
      // Determine MIME type from file extension
      const ext = path.extname(imagePath).toLowerCase();
      switch (ext) {
        case '.png':
          mimeType = 'image/png';
          break;
        case '.jpg':
        case '.jpeg':
          mimeType = 'image/jpeg';
          break;
        case '.gif':
          mimeType = 'image/gif';
          break;
        case '.webp':
          mimeType = 'image/webp';
          break;
        default:
          mimeType = 'image/png'; // Default fallback
      }
    } else {
      throw new Error("Either imageData or imagePath must be provided");
    }

    // Create the image part for input
    const imagePart = {
      inlineData: {
        data: finalImageData,
        mimeType: mimeType,
      },
    };

    // Build generation config for analysis
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
      content: [
        {
          type: "text",
          text: result.response.text(),
        },
      ],
    };
  }

  private async handleMultiImageEdit(args: unknown) {
    const { prompt, images, outputDir = ".", config } = MultiImageEditArgsSchema.parse(args);

    const model = this.genAI.getGenerativeModel({ 
      model: "gemini-2.5-flash-image-preview" 
    });

    // Process all images and create image parts
    const imageParts = [];
    for (let i = 0; i < images.length; i++) {
      const image = images[i];
      let finalImageData: string;
      let mimeType: string;

      if (image.imageData) {
        finalImageData = image.imageData;
        mimeType = "image/png"; // Default assumption
      } else if (image.imagePath) {
        // Read the image file and convert to base64
        const imageBuffer = await fs.readFile(image.imagePath);
        finalImageData = imageBuffer.toString('base64');
        
        // Determine MIME type from file extension
        const ext = path.extname(image.imagePath).toLowerCase();
        switch (ext) {
          case '.png':
            mimeType = 'image/png';
            break;
          case '.jpg':
          case '.jpeg':
            mimeType = 'image/jpeg';
            break;
          case '.gif':
            mimeType = 'image/gif';
            break;
          case '.webp':
            mimeType = 'image/webp';
            break;
          default:
            mimeType = 'image/png'; // Default fallback
        }
      } else {
        throw new Error(`Image ${i + 1} must have either imageData or imagePath`);
      }

      imageParts.push({
        inlineData: {
          data: finalImageData,
          mimeType: mimeType,
        },
      });
    }

    // Build generation config
    const generationConfig = config ? {
      temperature: config.temperature,
      topP: config.topP,
      topK: config.topK,
      maxOutputTokens: config.maxOutputTokens,
      responseModalities: ["TEXT", "IMAGE"],
    } : {
      responseModalities: ["TEXT", "IMAGE"],
    };

    // Build the content array with prompt and all images
    const result = await model.generateContent({
      contents: [{ role: "user", parts: [{ text: prompt }, ...imageParts] }],
      generationConfig,
    });
    
    if (!result.response.candidates || result.response.candidates.length === 0) {
      throw new Error("No multi-image result was generated");
    }

    const candidate = result.response.candidates[0];
    if (!candidate.content.parts || candidate.content.parts.length === 0) {
      throw new Error("No content parts in the response");
    }

    // Find the image part in the response
    const outputImagePart = candidate.content.parts.find(part => 'inlineData' in part);
    
    if (!outputImagePart || !('inlineData' in outputImagePart)) {
      throw new Error("No image data found in the response");
    }

    const outputImageData = outputImagePart.inlineData!.data;
    const outputMimeType = outputImagePart.inlineData!.mimeType;
    
    // Determine file extension from MIME type
    const extension = outputMimeType === 'image/png' ? '.png' : 
                     outputMimeType === 'image/jpeg' ? '.jpg' : 
                     '.png'; // default to PNG

    // Create filename with timestamp
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const filename = `multi-image-result-${timestamp}${extension}`;
    const filepath = path.join(outputDir, filename);

    // Save the result image
    await fs.mkdir(outputDir, { recursive: true });
    await fs.writeFile(filepath, Buffer.from(outputImageData, 'base64'));

    return {
      content: [
        {
          type: "text",
          text: `Multi-image processing completed successfully and saved to: ${filepath}`,
        },
      ],
    };
  }

  private async handleBatchGenerate(args: unknown) {
    const { prompts, outputDir = ".", config, parallel = false } = BatchGenerateArgsSchema.parse(args);

    const model = this.genAI.getGenerativeModel({ 
      model: "gemini-2.5-flash-image-preview" 
    });

    const generationConfig = config ? {
      temperature: config.temperature,
      topP: config.topP,
      topK: config.topK,
      maxOutputTokens: config.maxOutputTokens,
      responseModalities: ["TEXT", "IMAGE"],
    } : {
      responseModalities: ["TEXT", "IMAGE"],
    };

    const results = [];
    
    if (parallel) {
      // Process in parallel
      const promises = prompts.map(async (prompt, index) => {
        try {
          const result = await model.generateContent({
            contents: [{ role: "user", parts: [{ text: prompt }] }],
            generationConfig,
          });
          
          if (result.response.candidates?.[0]?.content.parts) {
            const imagePart = result.response.candidates[0].content.parts.find(part => 'inlineData' in part);
            if (imagePart && 'inlineData' in imagePart) {
              const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
              const filename = `batch-${index + 1}-${timestamp}.png`;
              const filepath = path.join(outputDir, filename);
              
              await fs.mkdir(outputDir, { recursive: true });
              await fs.writeFile(filepath, Buffer.from(imagePart.inlineData!.data, 'base64'));
              
              return { success: true, prompt, filepath };
            }
          }
          return { success: false, prompt, error: "No image generated" };
        } catch (error) {
          return { success: false, prompt, error: error instanceof Error ? error.message : String(error) };
        }
      });
      
      results.push(...await Promise.all(promises));
    } else {
      // Process sequentially
      for (let i = 0; i < prompts.length; i++) {
        try {
          const result = await model.generateContent({
            contents: [{ role: "user", parts: [{ text: prompts[i] }] }],
            generationConfig,
          });
          
          if (result.response.candidates?.[0]?.content.parts) {
            const imagePart = result.response.candidates[0].content.parts.find(part => 'inlineData' in part);
            if (imagePart && 'inlineData' in imagePart) {
              const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
              const filename = `batch-${i + 1}-${timestamp}.png`;
              const filepath = path.join(outputDir, filename);
              
              await fs.mkdir(outputDir, { recursive: true });
              await fs.writeFile(filepath, Buffer.from(imagePart.inlineData!.data, 'base64'));
              
              results.push({ success: true, prompt: prompts[i], filepath });
            } else {
              results.push({ success: false, prompt: prompts[i], error: "No image generated" });
            }
          }
        } catch (error) {
          results.push({ 
            success: false, 
            prompt: prompts[i], 
            error: error instanceof Error ? error.message : String(error) 
          });
        }
      }
    }

    const successful = results.filter(r => r.success).length;
    const failed = results.filter(r => !r.success).length;

    return {
      content: [
        {
          type: "text",
          text: `Batch generation completed: ${successful} successful, ${failed} failed\n` +
                results.map(r => r.success ? `✓ ${r.filepath}` : `✗ ${r.prompt}: ${r.error}`).join('\n'),
        },
      ],
    };
  }

  private async handleGenerateVariations(args: unknown) {
    const { imagePath, imageData, count = 3, variationStrength = "moderate", outputDir = ".", config } = GenerateVariationsArgsSchema.parse(args);

    const model = this.genAI.getGenerativeModel({ 
      model: "gemini-2.5-flash-image-preview" 
    });

    // Prepare base image
    let finalImageData: string;
    let mimeType: string;

    if (imageData) {
      finalImageData = imageData;
      mimeType = "image/png";
    } else if (imagePath) {
      const imageBuffer = await fs.readFile(imagePath);
      finalImageData = imageBuffer.toString('base64');
      const ext = path.extname(imagePath).toLowerCase();
      mimeType = ext === '.jpg' || ext === '.jpeg' ? 'image/jpeg' : 'image/png';
    } else {
      throw new Error("Either imageData or imagePath must be provided");
    }

    const imagePart = {
      inlineData: {
        data: finalImageData,
        mimeType: mimeType,
      },
    };

    // Variation prompts based on strength
    const variationPrompts = {
      subtle: "Create a very similar variation of this image with minimal changes, keeping the same style and composition",
      moderate: "Create a variation of this image with moderate changes while maintaining the core concept and style",
      strong: "Create a significantly different variation of this image, exploring new interpretations while keeping the main subject"
    };

    const basePrompt = variationPrompts[variationStrength];
    const generationConfig = config ? {
      temperature: config.temperature ?? (variationStrength === "subtle" ? 0.3 : variationStrength === "moderate" ? 0.7 : 1.2),
      topP: config.topP,
      topK: config.topK,
      maxOutputTokens: config.maxOutputTokens,
      responseModalities: ["TEXT", "IMAGE"],
    } : {
      temperature: variationStrength === "subtle" ? 0.3 : variationStrength === "moderate" ? 0.7 : 1.2,
      responseModalities: ["TEXT", "IMAGE"],
    };

    const results = [];
    
    for (let i = 0; i < count; i++) {
      try {
        const result = await model.generateContent({
          contents: [{ role: "user", parts: [{ text: `${basePrompt} (variation ${i + 1} of ${count})` }, imagePart] }],
          generationConfig,
        });
        
        if (result.response.candidates?.[0]?.content.parts) {
          const outputImagePart = result.response.candidates[0].content.parts.find(part => 'inlineData' in part);
          if (outputImagePart && 'inlineData' in outputImagePart) {
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
            const filename = `variation-${i + 1}-${timestamp}.png`;
            const filepath = path.join(outputDir, filename);
            
            await fs.mkdir(outputDir, { recursive: true });
            await fs.writeFile(filepath, Buffer.from(outputImagePart.inlineData!.data, 'base64'));
            
            results.push(filepath);
          }
        }
      } catch (error) {
        console.error(`Failed to generate variation ${i + 1}:`, error);
      }
    }

    return {
      content: [
        {
          type: "text",
          text: `Generated ${results.length} variations:\n${results.map(f => `- ${f}`).join('\n')}`,
        },
      ],
    };
  }

  private async handleGenerateWithTemplate(args: unknown) {
    const { template, customization, outputDir = ".", config } = PromptTemplateArgsSchema.parse(args);

    const model = this.genAI.getGenerativeModel({ 
      model: "gemini-2.5-flash-image-preview" 
    });

    // Combine template with customization
    const templatePrompt = this.promptTemplates[template];
    const fullPrompt = `${templatePrompt}. ${customization}`;

    const generationConfig = config ? {
      temperature: config.temperature,
      topP: config.topP,
      topK: config.topK,
      maxOutputTokens: config.maxOutputTokens,
      responseModalities: ["TEXT", "IMAGE"],
    } : {
      responseModalities: ["TEXT", "IMAGE"],
    };

    const result = await model.generateContent({
      contents: [{ role: "user", parts: [{ text: fullPrompt }] }],
      generationConfig,
    });
    
    if (!result.response.candidates || result.response.candidates.length === 0) {
      throw new Error("No image was generated");
    }

    const candidate = result.response.candidates[0];
    if (!candidate.content.parts || candidate.content.parts.length === 0) {
      throw new Error("No content parts in the response");
    }

    const imagePart = candidate.content.parts.find(part => 'inlineData' in part);
    
    if (!imagePart || !('inlineData' in imagePart)) {
      throw new Error("No image data found in the response");
    }

    const imageData = imagePart.inlineData!.data;
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const filename = `${template}-${timestamp}.png`;
    const filepath = path.join(outputDir, filename);

    await fs.mkdir(outputDir, { recursive: true });
    await fs.writeFile(filepath, Buffer.from(imageData, 'base64'));

    return {
      content: [
        {
          type: "text",
          text: `Generated ${template} style image: ${filepath}\nPrompt used: ${fullPrompt}`,
        },
      ],
    };
  }

  private async handleCompareImages(args: unknown) {
    const { image1Path, image2Path, compareType = "both" } = CompareImagesArgsSchema.parse(args);

    const model = this.genAI.getGenerativeModel({ 
      model: "gemini-2.5-flash"
    });

    // Read both images
    const image1Buffer = await fs.readFile(image1Path);
    const image2Buffer = await fs.readFile(image2Path);

    const image1Part = {
      inlineData: {
        data: image1Buffer.toString('base64'),
        mimeType: 'image/png',
      },
    };

    const image2Part = {
      inlineData: {
        data: image2Buffer.toString('base64'),
        mimeType: 'image/png',
      },
    };

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
      content: [
        {
          type: "text",
          text: `Image Comparison (${compareType}):\n\n${result.response.text()}`,
        },
      ],
    };
  }

  async run() {
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
    console.error("Nano Banana MCP Server running on stdio");
  }
}

const server = new NanaBananaMCPServer();
server.run().catch(console.error);