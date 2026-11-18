"""Source mapping and local dependencies for browser entrypoints."""
from dataclasses import dataclass
from html.parser import HTMLParser
import posixpath
import re
from urllib.parse import unquote, urlsplit


HTML_SUFFIXES = {'.html', '.htm'}
FRONTEND_SUFFIXES = HTML_SUFFIXES | {'.css'}
_JS_TYPES = {'', 'module', 'text/javascript', 'application/javascript', 'text/ecmascript',
             'application/ecmascript', 'text/jscript', 'text/x-javascript', 'application/x-javascript'}
_CSS_TOKEN = re.compile(
    r"""/\*[\s\S]*?\*/|"([^"\\]|\\.)*"|'([^'\\]|\\.)*'"""
    r"""|@import\s+(?P<import>"[^"\n]*"|'[^'\n]*')"""
    r"""|\burl\(\s*(?P<url>"[^"\n]*"|'[^'\n]*'|[^\s)]+)\s*\)""", re.IGNORECASE)


@dataclass(frozen=True)
class SourceBlock:
    language: str
    source: str
    start_line: int


@dataclass(frozen=True)
class Reference:
    kind: str
    value: str
    line: int


def css_references(source, *, start_line=1):
    for match in _CSS_TOKEN.finditer(source):
        value = match['import'] or match['url']
        if value:
            yield Reference('css_import' if match['import'] else 'css_url',
                            value.strip("'\""), start_line + source.count('\n', 0, match.start()))


class _DocumentParser(HTMLParser):
    def __init__(self, source):
        super().__init__(convert_charrefs=False)
        self.source = source
        self.line_offsets = [0, *(match.end() for match in re.finditer('\n', source))]
        self.blocks = []
        self.references = []
        self.current_block = None

    def handle_starttag(self, tag, attrs):
        self._tag(tag, attrs, closed=False)

    def handle_startendtag(self, tag, attrs):
        self._tag(tag, attrs, closed=True)

    def _tag(self, tag, attrs, *, closed):
        # The browser uses the first instance of a duplicate attribute.
        attributes = dict(reversed(attrs))
        line, column = self.getpos()
        mime = (attributes.get('type') or '').split(';')[0].strip().lower()
        if tag == 'script' and mime in _JS_TYPES:
            if 'src' in attributes:
                self._reference('script', attributes['src'], line)
            elif not closed:
                self._begin('javascript', line, column)
        elif tag == 'style' and mime in {'', 'text/css'} and not closed:
            self._begin('css', line, column)
        elif tag == 'link' and 'stylesheet' in (attributes.get('rel') or '').lower().split():
            self._reference('stylesheet', attributes.get('href'), line)
        if attributes.get('style'):
            self.references.extend(css_references(attributes['style'], start_line=line))

    def _reference(self, kind, value, line):
        if value and value.strip():
            self.references.append(Reference(kind, value.strip(), line))

    def _begin(self, language, line, column):
        tag = self.get_starttag_text()
        begin = self.line_offsets[line - 1] + column + len(tag)
        self.current_block = (language, begin, line + tag.count('\n'))

    def handle_endtag(self, tag):
        if tag in {'script', 'style'} and self.current_block:
            line, column = self.getpos()
            language, begin, start_line = self.current_block
            content = self.source[begin:self.line_offsets[line - 1] + column]
            self.blocks.append(SourceBlock(language, content, start_line))
            if language == 'css':
                self.references.extend(css_references(content, start_line=start_line))
            self.current_block = None


def parse_html(source):
    parser = _DocumentParser(source)
    parser.feed(source)
    parser.close()
    return tuple(parser.blocks), tuple(sorted(parser.references, key=lambda reference: reference.line))


def local_reference(file, value):
    """Resolve a browser URL against a repository path without reading the network."""
    try:
        url = urlsplit(value.strip())
    except ValueError:
        return None
    if url.scheme or url.netloc or not url.path:
        return None
    path = unquote(url.path)
    if '\\' in path or '\0' in path:
        return None
    if path.startswith('/'):
        base = path.lstrip('/')
    else:
        base = posixpath.join(posixpath.dirname(file), path)
    normalized = posixpath.normpath(base)
    return normalized if normalized not in {'.', '..'} and not normalized.startswith('../') else None
