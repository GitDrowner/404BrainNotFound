# RobustFusion

TikTok TechJam 2026 Track 5 展示前端。模型全称为 **RobustFusion: Robust AI-Generated Image Detection via Multi-Cue Fusion**。页面聚焦交互式图片变换、局部 patch 贡献、16 条件置信度轨迹与分支反事实贡献。

> 当前页面已接入 `773086` FastAPI：上传图片会执行真实模型推理、渐进式 16 变换扫描，
> 并可生成模型归因图与分支反事实贡献。文件只发送到本机后端。

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

推荐从仓库根目录运行：

```bash
./scripts/setup_demo.sh
./scripts/run_demo.sh
```

也可单独启动前端：

```bash
corepack enable
pnpm install
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000 pnpm dev
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

## 后端契约

- `GET /api/health`：连通性与设备状态
- `GET /api/v1/transforms`：后端定义的 16 项变换目录
- `POST /api/v1/predict`：当前变换同步推理并创建后台扫描
- `GET /api/v1/transform-scans/{scan_id}`：渐进式扫描轮询
- `POST /api/v1/analyses`：创建真实解释任务
- `GET /api/v1/analyses/{job_id}`：解释任务轮询

完整字段见 [`../773086/BACKEND_API.md`](../773086/BACKEND_API.md)。
