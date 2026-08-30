# RobustFusion

TikTok TechJam 2026 Track 5 展示前端。模型全称为 **RobustFusion: Robust AI-Generated Image Detection via Multi-Cue Fusion**。页面聚焦交互式图片变换、局部 patch 贡献、16 条件置信度轨迹与分支反事实贡献。

> 当前上传与检测流程是浏览器端交互原型：图片只在本地生成预览，不会上传，也尚未调用真实模型推理服务。

## 技术栈

- React 19 + TypeScript
- Next.js 兼容 App Router API
- Vinext + Vite 8
- Tailwind CSS 4（基础加载）+ 手写响应式 CSS
- OpenAI Sites Vite Plugin
- Cloudflare Vite Plugin / Wrangler 部署运行时

## PPT 同款视觉令牌

- `#080806`：主背景 / 近黑
- `#25F4EE`：TikTok 青 / 主强调色
- `#FE2C55`：TikTok 洋红 / 风险与对比强调
- `#70D0D8`：柔和青 / 次级信息层
- `#F7F7F7`：正文与浅色背景
- `#A8ADB3`：说明文字
- `#242428`：分隔线与深色卡片边界

## 环境要求

- Node.js 22.13 或更高版本
- pnpm 10 或更高版本

## 安装与运行

```bash
corepack enable
pnpm install
pnpm dev
```

开发服务器启动后访问：

```text
http://localhost:3000
```

## 生产构建

```bash
pnpm build
pnpm start
```

## 代码检查

```bash
pnpm lint
```

## 核心文件

- `app/page.tsx`：页面内容、交互状态、上传预览、变换切换和指标图表
- `app/globals.css`：视觉系统、布局、动画和响应式样式
- `app/layout.tsx`：页面元信息与社交分享卡片配置
- `public/og.png`：社交分享图
- `vite.config.ts`：Vinext、Sites 与 Cloudflare 构建配置
- `.openai/hosting.json`：Sites 部署项目配置

## 接入真实模型

将 `app/page.tsx` 中的 `onFile` 演示逻辑替换为真实推理请求。推荐接口接收图片文件并返回：

```json
{
  "pred": 0.978,
  "label": "ai_generated",
  "signals": {
    "frequency_artifacts": "high",
    "semantic_consistency": "medium",
    "transformation_resilience": "pass"
  }
}
```
