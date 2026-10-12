import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import { unified } from '@astrojs/markdown-remark';
import rehypeFigure from './src/lib/rehype-figure.mjs';
import rehypeSidenotes from './src/lib/rehype-sidenotes.mjs';

// On GitHub Pages these come from the deploy workflow, so the site works
// both at username.github.io and at username.github.io/repo-name.
export default defineConfig({
  site: process.env.SITE_ORIGIN || 'http://localhost:4321',
  base: process.env.SITE_BASE || '/',
  integrations: [mdx()],
  markdown: {
    processor: unified({ rehypePlugins: [rehypeFigure, rehypeSidenotes] }),
    syntaxHighlight: false,
  },
});
