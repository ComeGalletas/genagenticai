import { useRef, useMemo, useState, useEffect, Fragment } from "react";
import DOMPurify from "dompurify";

// Make every link inside sanitized HTML open in a new tab safely.
DOMPurify.addHook("afterSanitizeAttributes", (node) => {
  if (node.tagName === "A") {
    node.setAttribute("target", "_blank");
    node.setAttribute("rel", "noreferrer noopener");
  }
});

const HTML_TAG_REGEX = /<\/?[a-zA-Z][^>]*>/;
const URL_SPLIT_REGEX = /(https?:\/\/[^\s<>"')\]]+)/g;
const URL_TEST_REGEX = /^https?:\/\//

function renderWithLinks(text) {
  const parts = text.split(URL_SPLIT_REGEX);
  return parts.map((part, i) =>
    URL_TEST_REGEX.test(part) ? (
      <a key={i} href={part} target="_blank" rel="noreferrer noopener">
        &lt;&lt;View Link&gt;&gt;
      </a>
    ) : (
      <Fragment key={i}>{part}</Fragment>
    )
  );
}

function MessageContent({ content, streaming, status }) {
  if (streaming) {
    // Draft: raw markdown tokens as they arrive, swapped for sanitized HTML on the final event.
    return (
      <>
        {content ? <p className="draft">{content}</p> : null}
        {status ? <p className="status">{status}</p> : null}
      </>
    );
  }
  if (HTML_TAG_REGEX.test(content)) {
    const clean = DOMPurify.sanitize(content);
    return <div className="html-content" dangerouslySetInnerHTML={{ __html: clean }} />;
  }
  return <p>{renderWithLinks(content)}</p>;
}

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

/** Parse a fetch body as Server-Sent Events, calling onEvent({event, ...data}) per frame. */
async function readSse(response, onEvent) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let separator;
    while ((separator = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, separator);
      buffer = buffer.slice(separator + 2);
      const dataLine = frame.split("\n").find((line) => line.startsWith("data:"));
      if (!dataLine) continue;
      try {
        onEvent(JSON.parse(dataLine.slice(5).trim()));
      } catch {
        // malformed frame: ignore and keep reading
      }
    }
  }
}

/** Generate a random session ID that persists for the lifetime of this page load. */
function generateThreadId() {
  return crypto.randomUUID();
}

function App() {
  // Each browser session gets its own thread_id so MemorySaver separates histories.
  const threadId = useRef(generateThreadId());
  const bottomRef = useRef(null);

  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content:
        "Hi, I am your AI assistant. Ask me to search anything and I will try my best.",
    },
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  // "checking" | "online" | "offline"
  const [backendStatus, setBackendStatus] = useState("checking");

  // Check backend health on mount, then retry every 10s until it answers.
  useEffect(() => {
    let timer;
    const check = () =>
      fetch(`${API_BASE_URL}/health`)
        .then((res) => res.ok)
        .catch(() => false)
        .then((ok) => {
          setBackendStatus(ok ? "online" : "offline");
          if (ok) clearInterval(timer);
        });

    check();
    timer = setInterval(check, 10000);
    return () => clearInterval(timer);
  }, []);

  // Scroll to the bottom whenever a message is added or the loading indicator appears.
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  const disabled = useMemo(
    () => isLoading || input.trim().length === 0,
    [isLoading, input]
  );

  async function handleSubmit(event) {
    event.preventDefault();
    const content = input.trim();
    if (!content) return;

    setMessages((prev) => [...prev, { role: "user", content }]);
    setInput("");
    setIsLoading(true);

    const body = JSON.stringify({ message: content, thread_id: threadId.current });
    const headers = { "Content-Type": "application/json" };

    // Draft assistant message that the stream fills in; replaced by the final HTML.
    let draftIndex = -1;
    setMessages((prev) => {
      draftIndex = prev.length;
      return [...prev, { role: "assistant", content: "", streaming: true, status: "Thinking…" }];
    });
    const updateDraft = (patch) =>
      setMessages((prev) => prev.map((m, i) => (i === draftIndex ? { ...m, ...patch } : m)));

    try {
      let finalHtml = null;
      try {
        const response = await fetch(`${API_BASE_URL}/api/chat/stream`, { method: "POST", headers, body });
        if (!response.ok || !response.body) {
          throw new Error(`Stream request failed with ${response.status}`);
        }
        await readSse(response, (event) => {
          switch (event.event) {
            case "status":
              updateDraft({ status: event.text });
              break;
            case "delta":
              setMessages((prev) =>
                prev.map((m, i) => (i === draftIndex ? { ...m, content: m.content + event.text, status: "" } : m))
              );
              break;
            case "reset":
              updateDraft({ content: "" });
              break;
            case "final":
              finalHtml = event.html;
              break;
            case "error":
              throw new Error(event.message);
            default:
              break;
          }
        });
      } catch (streamError) {
        // Streaming unavailable (older backend, proxy buffering, network): fall back to one-shot.
        console.warn("Streaming failed, falling back to /api/chat:", streamError);
        updateDraft({ content: "", status: "Thinking…" });
        const response = await fetch(`${API_BASE_URL}/api/chat`, { method: "POST", headers, body });
        if (!response.ok) {
          throw new Error(`Request failed with ${response.status}`);
        }
        finalHtml = (await response.json()).reply;
      }
      updateDraft({ content: finalHtml ?? "", streaming: false, status: "" });
    } catch (error) {
      updateDraft({ content: `I hit an error: ${error.message}`, streaming: false, status: "" });
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <main className="page">
      <section className="chat-shell">
        <header className="chat-header">
          <p className="eyebrow">Agentic AI Starter</p>
          <div className="header-row">
            <h1>LangGraph + React Chat</h1>
            <span
              className={`status-dot status-${backendStatus}`}
              title={`Backend: ${backendStatus}`}
              aria-label={`Backend status: ${backendStatus}`}
            />
          </div>
        </header>

        <div className="messages">
          {messages.map((message, index) => (
            <article
              key={`${message.role}-${index}`}
              className={`message ${message.role}${message.streaming ? " loading" : ""}`}
            >
              <MessageContent content={message.content} streaming={message.streaming} status={message.status} />
            </article>
          ))}
          <div ref={bottomRef} />
        </div>

        <form className="composer" onSubmit={handleSubmit}>
          <input
            value={input}
            onChange={(event) => setInput(event.target.value)}
            placeholder="Try: Search LangGraph docs"
            aria-label="Message"
          />
          <button type="submit" disabled={disabled}>
            Send
          </button>
        </form>
      </section>
    </main>
  );
}

export default App;
