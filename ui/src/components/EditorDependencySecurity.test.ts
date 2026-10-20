import { describe, expect, it } from 'vitest';
import { Editor, mergeAttributes } from '@tiptap/core';
import StarterKit from '@tiptap/starter-kit';
import TaskList from '@tiptap/extension-task-list';
import TaskItem from '@tiptap/extension-task-item';
import { Markdown } from 'tiptap-markdown';
import { Schema, type ResolvedPos, type Slice } from '@tiptap/pm/model';
import { EditorState } from '@tiptap/pm/state';
import { EditorView } from '@tiptap/pm/view';
import * as viewModule from '@tiptap/pm/view';

describe('editor dependency security', () => {
  it('does not inherit DOM attributes from an own __proto__ property', () => {
    // GHSA-cp6q-959q-f8rh: use inert canaries, never executable payloads.
    const input = JSON.parse('{"__proto__":{"data-inherited-canary":"unexpected"}}');
    const attributes = mergeAttributes({ class: 'safe' }, input);
    expect(attributes.class).toBe('safe');
    expect('data-inherited-canary' in attributes).toBe(false);
  });

  it('validates attributes from pasted ProseMirror slice context', () => {
    // GHSA-c8x8-7fp4-3x9w. The parser is an upstream-exported test hook.
    const checked: unknown[] = [];
    const schema = new Schema({
      nodes: {
        doc: { content: 'block+' },
        paragraph: {
          group: 'block', content: 'inline*',
          attrs: { title: { default: null, validate(value: unknown) {
            checked.push(value);
            if (value !== null && typeof value !== 'string') throw new RangeError('invalid title');
          } } },
          parseDOM: [{ tag: 'p' }],
          toDOM: () => ['p', 0],
        },
        text: { group: 'inline' },
      },
    });
    const view = new EditorView(document.createElement('div'), {
      state: EditorState.create({ schema }),
    });
    try {
      const parseClipboard = (viewModule as unknown as {
        __parseFromClipboard: (view: EditorView, text: string, html: string,
          plainText: boolean, context: ResolvedPos) => Slice | null;
      }).__parseFromClipboard;
      const html = `<p data-pm-slice='1 1 ["paragraph",{"title":{"probe":"invalid"}}]'>hello</p>`;
      const slice = parseClipboard(view, '', html, false, view.state.selection.$from);
      expect(checked).toContainEqual({ probe: 'invalid' });
      expect(slice?.content.textBetween(0, slice.content.size)).toBe('hello');
      slice?.content.descendants(node => {
        expect(node.attrs.title).not.toEqual({ probe: 'invalid' });
      });
    } finally {
      view.destroy();
    }
  });

  it('keeps headings, links, and task lists in the configured Markdown editor', () => {
    const editor = new Editor({
      extensions: [
        StarterKit.configure({ link: { openOnClick: false, autolink: true, linkOnPaste: true } }),
        TaskList,
        TaskItem.configure({ nested: true }),
        Markdown.configure({ html: false, tightLists: true, linkify: true,
          breaks: false, transformPastedText: true, transformCopiedText: true }),
      ],
      content: '# Controle\n\n[Veilige link](https://example.com)\n\n- [x] Gereed\n- [ ] Open',
    });
    try {
      const html = editor.getHTML();
      expect(html).toContain('<h1>Controle</h1>');
      expect(html).toContain('href="https://example.com"');
      expect(html).toContain('data-type="taskList"');
      expect(html).toContain('data-checked="true"');
      expect(html).toContain('data-checked="false"');
      const storage = editor.storage as unknown as { markdown: { getMarkdown: () => string } };
      expect(storage.markdown.getMarkdown()).toContain('[Veilige link](https://example.com)');
      expect(storage.markdown.getMarkdown()).toContain('[x] Gereed');
    } finally {
      editor.destroy();
    }
  });
});
