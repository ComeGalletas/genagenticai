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

function MessageContent({ content }) {
  if (HTML_TAG_REGEX.test(content)) {
    const clean = DOMPurify.sanitize(content);
    return <div className="html-content" dangerouslySetInnerHTML={{ __html: clean }} />;
  }
  return <p>{renderWithLinks(content)}</p>;
}

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

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

  // Check backend health once on mount.
  useEffect(() => {
    fetch(`${API_BASE_URL}/health`)
      .then((res) => (res.ok ? setBackendStatus("online") : setBackendStatus("offline")))
      .catch(() => setBackendStatus("offline"));
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

    const nextMessages = [...messages, { role: "user", content }];
    setMessages(nextMessages);
    setInput("");
    setIsLoading(true);

    try {
      const response = await fetch(`${API_BASE_URL}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: content, thread_id: threadId.current }),
      });

      if (!response.ok) {
        throw new Error(`Request failed with ${response.status}`);
      }

      const data = await response.json();
      setMessages((prev) => [...prev, { role: "assistant", content: data.reply }]);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: `I hit an error: ${error.message}`,
        },
      ]);
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
            <article key={`${message.role}-${index}`} className={`message ${message.role}`}>
              <MessageContent content={message.content} />
            </article>
          ))}
          {isLoading && (
            <article className="message assistant loading">
              <p>Thinking...</p>
            </article>
          )}
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
