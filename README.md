# 🎨 Nano Banana MCP Server

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MCP](https://img.shields.io/badge/MCP-Compatible-green.svg)](https://modelcontextprotocol.io)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-blue.svg)](https://www.typescriptlang.org/)

A powerful Model Context Protocol (MCP) server for advanced image generation, editing, and analysis using Google's Gemini 2.5 Flash Image Preview (aka "nano-banana") and Gemini 3 Pro Image Preview (aka "nano-banana pro") models.

**English** | [日本語](README.ja.md)

## ✨ Features

### Core Capabilities
- 🖼️ **Image Generation** - Create images from text prompts with advanced customization
- ✏️ **Image Editing** - Edit existing images using natural language (supports file paths or base64)
- 🔍 **Image Analysis** - Analyze and answer questions about images using Gemini
- 🎭 **Multi-Image Processing** - Combine, style transfer, or edit multiple images at once
- 📦 **Batch Generation** - Generate multiple images from an array of prompts
- 🎨 **Style Templates** - Use pre-defined templates for consistent styling
- 🔄 **Variations** - Generate variations of existing images with controlled randomness
- 🔬 **Image Comparison** - Compare and analyze differences between images

### Advanced Features
- **Dual Model Support** - Choose between Gemini 2.5 Flash Image Preview or Gemini 3 Pro Image Preview
- **High Resolution Output** - Generate up to 4K images (4096x4096) with Gemini 3 Pro
- **Generation Control** - Fine-tune with temperature, topP, topK parameters
- **Parallel Processing** - Batch operations with optional parallel execution
- **Smart Templates** - 10+ professional style presets
- **Flexible Input** - Accept both file paths and base64 encoded images
- **Auto-save** - Automatically saves generated images with timestamps

### Model Comparison

| Feature | Gemini 2.5 Flash Image | Gemini 3 Pro Image |
|---------|------------------------|-------------------|
| Max Resolution | 1K (1024px) | 1K / 2K / 4K |
| Aspect Ratios | All supported | All supported |
| Text Rendering | Good | Excellent |
| Speed | Fast | Moderate |
| Quality | High | Premium |
| Best For | Quick iterations | Final production |

**Supported Aspect Ratios**: `1:1`, `2:3`, `3:2`, `3:4`, `4:3`, `4:5`, `5:4`, `9:16`, `16:9`, `21:9`

## 🚀 Quick Start

### Prerequisites
- Node.js 18.0.0 or higher
- Valid [Gemini API key](https://aistudio.google.com/)
- MCP-compatible client (like [Claude Code](https://claude.ai/code))

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Riti0208/nano-banana-mcp.git
cd nano-banana-mcp
```

2. Install dependencies:
```bash
npm install
```

3. Build the project:
```bash
npm run build
```

### Configuration

Add to your Claude Code MCP settings (`claude_code_config.json`):

```json
{
  "mcpServers": {
    "nano-banana": {
      "command": "node",
      "args": ["./dist/index.js"],
      "cwd": "/path/to/nano-banana-mcp",
      "env": {
        "GEMINI_API_KEY": "your-api-key-here"
      }
    }
  }
}
```

## 📖 Usage Examples

### Generate an Image (Standard)
```javascript
generate_image({
  prompt: "A serene mountain landscape at sunset",
  config: {
    temperature: 0.8,
    topP: 0.95
  }
})
```

### Generate a High-Resolution Image (4K with Gemini 3 Pro)
```javascript
generate_image({
  prompt: "Ultra detailed cyberpunk city at night with neon signs",
  model: "gemini-3-pro-image-preview",
  resolution: "4K",
  aspectRatio: "16:9",
  config: {
    temperature: 0.9
  }
})
```

### Edit an Image
```javascript
edit_image({
  prompt: "Add a rainbow in the sky",
  imagePath: "./landscape.jpg",
  model: "gemini-3-pro-image-preview",
  resolution: "2K",
  aspectRatio: "4:3",
  config: {
    temperature: 0.5
  }
})
```

### Generate Multiple Variations
```javascript
generate_variations({
  imagePath: "./original.png",
  count: 3,
  variationStrength: "moderate"
})
```

### Batch Generation
```javascript
batch_generate({
  prompts: [
    "A red apple",
    "A green apple",
    "A golden apple"
  ],
  parallel: true
})
```

### Use Style Templates
```javascript
generate_with_template({
  template: "photorealistic",
  customization: "A vintage coffee shop interior"
})
```

### Compare Images
```javascript
compare_images({
  image1Path: "./before.png",
  image2Path: "./after.png",
  compareType: "differences"
})
```

## 🛠️ Available Tools

| Tool | Description |
|------|-------------|
| `generate_image` | Generate images from text prompts |
| `edit_image` | Edit existing images with natural language |
| `analyze_image` | Analyze images and answer questions |
| `multi_image_edit` | Process multiple images together |
| `batch_generate` | Generate multiple images at once |
| `generate_variations` | Create variations of an image |
| `generate_with_template` | Use predefined style templates |
| `compare_images` | Compare two images |

## 🎨 Style Templates

- **photorealistic** - Ultra-realistic photography
- **artistic** - Artistic interpretation
- **logo** - Clean logo design
- **portrait** - Professional portraits
- **landscape** - Breathtaking landscapes
- **product** - Product photography
- **architectural** - Architectural photography
- **fashion** - Fashion photography
- **food** - Food photography
- **abstract** - Abstract art

## ⚙️ Configuration Parameters

| Parameter | Range | Description |
|-----------|-------|-------------|
| `temperature` | 0.0-2.0 | Controls randomness (lower = more focused) |
| `topP` | 0.0-1.0 | Nucleus sampling threshold |
| `topK` | 1-40 | Top-k sampling |
| `maxOutputTokens` | - | Maximum response length |

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Google Gemini team for the amazing image generation API
- Model Context Protocol team for the MCP framework
- All contributors who help improve this project

## 🔗 Links

- [Gemini API Documentation](https://ai.google.dev/gemini-api/docs)
- [Model Context Protocol](https://modelcontextprotocol.io)
- [Report Issues](https://github.com/Riti0208/nano-banana-mcp/issues)

