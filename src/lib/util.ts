import { getCollection } from 'astro:content';

const base = import.meta.env.BASE_URL.replace(/\/$/, '');

export const url = (path: string) => `${base}${path}`;

export const isoDate = (d: Date) => d.toISOString().slice(0, 10);

export const longDate = (d: Date) =>
  d.toLocaleDateString('en-GB', { day: 'numeric', month: 'long', year: 'numeric', timeZone: 'UTC' });

export async function getPosts() {
  const posts = await getCollection('writing', ({ data }) => import.meta.env.DEV || !data.draft);
  return posts.sort((a, b) => b.data.date.valueOf() - a.data.date.valueOf());
}

export async function getNotes() {
  const notes = await getCollection('notes');
  return notes.sort((a, b) => b.data.date.valueOf() - a.data.date.valueOf());
}
