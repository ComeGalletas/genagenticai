import unittest

from app.graph.core.formatting import render_reply, sanitize_html


class RenderReplyTests(unittest.TestCase):
    def test_markdown_basics_become_html(self) -> None:
        out = render_reply("# Title nya~\n\n**32 GB** of *GDDR7*\n\n- a\n- b\n\n1. one\n2. two")
        self.assertIn("<h1>Title nya~</h1>", out)
        self.assertIn("<strong>32 GB</strong>", out)
        self.assertIn("<em>GDDR7</em>", out)
        self.assertIn("<ul>", out)
        self.assertIn("<ol>", out)
        self.assertNotIn("**", out)

    def test_links_and_tables(self) -> None:
        out = render_reply("[docs](https://example.com/x)\n\n| a | b |\n|---|---|\n| 1 | 2 |")
        self.assertIn('<a href="https://example.com/x">docs</a>', out)
        self.assertIn("<table>", out)
        self.assertIn("<td>1</td>", out)

    def test_html_from_model_passes_through_sanitized(self) -> None:
        out = render_reply('<p>Hello <strong>there</strong> <span onclick="x()">nya~</span></p>')
        self.assertIn("<p>Hello <strong>there</strong> <span>nya~</span></p>", out)
        self.assertNotIn("onclick", out)

    def test_plain_text_is_wrapped_in_paragraph(self) -> None:
        self.assertEqual(render_reply("Hello nya~"), "<p>Hello nya~</p>")

    def test_empty_and_think_blocks(self) -> None:
        self.assertEqual(render_reply(""), "")
        self.assertEqual(render_reply("<think>secret plan</think>\n\nDone!"), "<p>Done!</p>")

    def test_fenced_code_keeps_language_class(self) -> None:
        out = render_reply("```python\nprint(1)\n```")
        self.assertIn('<pre><code class="language-python">print(1)', out)


class SanitizeHtmlTests(unittest.TestCase):
    def test_scripts_and_styles_are_dropped_with_content(self) -> None:
        out = sanitize_html("<p>ok</p><script>alert(1)</script><style>p{}</style><p>still ok</p>")
        self.assertEqual(out, "<p>ok</p><p>still ok</p>")

    def test_unknown_tags_are_unwrapped(self) -> None:
        self.assertEqual(sanitize_html("<custom><p>text</p></custom>"), "<p>text</p>")

    def test_unsafe_hrefs_are_removed(self) -> None:
        self.assertEqual(sanitize_html('<a href="javascript:alert(1)">x</a>'), "<a>x</a>")
        self.assertEqual(sanitize_html('<a href="https://ok.com" onclick="y()">x</a>'), '<a href="https://ok.com">x</a>')

    def test_unbalanced_tags_are_closed(self) -> None:
        self.assertEqual(sanitize_html("<p><strong>open"), "<p><strong>open</strong></p>")
        self.assertEqual(sanitize_html("</p>stray"), "stray")

    def test_text_is_escaped(self) -> None:
        self.assertEqual(sanitize_html("a < b & c"), "a &lt; b &amp; c")


if __name__ == "__main__":
    unittest.main()
