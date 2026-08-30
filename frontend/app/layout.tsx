import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  metadataBase: new URL('https://truepix-techjam-2026.hiiiyou222.chatgpt.site'),
  title: 'RobustFusion — Robust AI-Generated Image Detection',
  description: 'Robust AI-generated image detection via multi-cue fusion across compression, blur, resizing, noise, color edits and crops.',
  openGraph: {
    title: 'RobustFusion — Multi-Cue AI-Generated Image Detection',
    description: 'Robust evidence after the edit.',
    images: [{ url: '/og.png', width: 1200, height: 630, alt: 'RobustFusion AI-generated image detection' }],
  },
  twitter: {
    card: 'summary_large_image',
    title: 'RobustFusion — Multi-Cue AI-Generated Image Detection',
    description: 'Robust evidence after the edit.',
    images: ['/og.png'],
  },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
