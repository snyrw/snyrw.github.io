import rss from '@astrojs/rss';
import { site } from '../site';
import { getPosts } from '../lib/util';

export async function GET(context) {
  const posts = await getPosts();
  const base = import.meta.env.BASE_URL.replace(/\/$/, '');
  return rss({
    title: site.name,
    description: site.description,
    site: new URL(base + '/', context.site),
    items: posts.map((post) => ({
      title: post.data.title,
      pubDate: post.data.date,
      description: post.data.description,
      link: `${base}/writing/${post.id}/`,
    })),
  });
}
