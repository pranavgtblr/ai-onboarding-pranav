import React, { useState, useEffect, useRef } from 'react';
import { 
  Clapperboard, 
  Sparkles, 
  Send, 
  RefreshCw, 
  AlertCircle, 
  Film, 
  ExternalLink, 
  UserCheck, 
  Heart, 
  ThumbsDown, 
  Compass, 
  MessageSquareShare,
  X
} from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

export default function App() {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      text: "Hey! Pranav here. Tell me what kind of mood you're in, what you've watched recently, or what genres you want to explore, and I'll dig into my Letterboxd diary to hook you up with something genuinely great. What are you feeling today?",
      citations: []
    }
  ]);
  const [input, setInput] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [tasteProfile, setTasteProfile] = useState({
    liked_directors: ['Denis Villeneuve', 'Christopher Nolan', 'Joel Coen'],
    liked_genres: ['Psychological Thriller', 'Horror', 'Neo-Noir'],
    disliked_elements: ['Jump scares', 'Cringe dialogue montages'],
    mood_tags: ['Atmospheric']
  });
  const [recentWatches, setRecentWatches] = useState([]);
  const [isSyncing, setIsSyncing] = useState(false);
  const [isEscalateOpen, setIsEscalateOpen] = useState(false);
  const [escalateReason, setEscalateReason] = useState('');
  const [escalationTicket, setEscalationTicket] = useState(null);

  const messagesEndRef = useRef(null);

  // Auto-scroll to bottom of chat
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Load initial catalog & taste profile
  useEffect(() => {
    fetch(`${API_BASE}/api/movies?limit=8`)
      .then(res => res.json())
      .then(data => {
        if (data.movies) setRecentWatches(data.movies);
      })
      .catch(err => console.log("Backend offline or loading:", err));

    fetch(`${API_BASE}/api/taste-profile?tenant_id=default_tenant&user_id=guest_user`)
      .then(res => res.json())
      .then(data => {
        if (data.liked_genres?.length > 0 || data.liked_directors?.length > 0) {
          setTasteProfile(data);
        }
      })
      .catch(() => {});
  }, []);

  const handleSend = async (messageText = input) => {
    const trimmed = messageText.trim();
    if (!trimmed || isStreaming) return;

    setInput('');
    const userMsg = { role: 'user', text: trimmed };
    setMessages(prev => [...prev, userMsg]);
    setIsStreaming(true);

    const assistantMsgIndex = messages.length + 1;
    setMessages(prev => [
      ...prev,
      { role: 'assistant', text: '', citations: [], isStreaming: true }
    ]);

    try {
      const response = await fetch(`${API_BASE}/api/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          tenant_id: 'default_tenant',
          user_id: 'guest_user',
          conversation_id: 'conv_pg_session',
          message: trimmed
        })
      });

      if (!response.body) throw new Error("No response body");

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let streamedText = '';
      let citations = [];
      let criticCitations = [];

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const dataStr = line.replace('data: ', '').trim();
            if (!dataStr) continue;

            try {
              const eventData = JSON.parse(dataStr);

              if (eventData.event === 'token') {
                streamedText += eventData.data;
                setMessages(prev => {
                  const updated = [...prev];
                  updated[updated.length - 1] = {
                    role: 'assistant',
                    text: streamedText,
                    citations: citations,
                    criticCitations: criticCitations,
                    isStreaming: true
                  };
                  return updated;
                });
              } else if (eventData.event === 'citations') {
                citations = eventData.data;
                setMessages(prev => {
                  const updated = [...prev];
                  updated[updated.length - 1] = {
                    role: 'assistant',
                    text: streamedText,
                    citations: citations,
                    criticCitations: criticCitations,
                    isStreaming: true
                  };
                  return updated;
                });
              } else if (eventData.event === 'critic_citations') {
                criticCitations = eventData.data;
                setMessages(prev => {
                  const updated = [...prev];
                  updated[updated.length - 1] = {
                    role: 'assistant',
                    text: streamedText,
                    citations: citations,
                    criticCitations: criticCitations,
                    isStreaming: true
                  };
                  return updated;
                });
              } else if (eventData.event === 'taste_update') {
                setTasteProfile(prev => ({
                  ...prev,
                  ...eventData.data
                }));
              } else if (eventData.event === 'escalation') {
                setEscalationTicket(eventData.data.ticket_id);
              }
            } catch (e) {
              // Ignore partial JSON chunks
            }
          }
        }
      }

      setMessages(prev => {
        const updated = [...prev];
        updated[updated.length - 1] = {
          ...updated[updated.length - 1],
          isStreaming: false,
          citations: citations,
          criticCitations: criticCitations
        };
        return updated;
      });
    } catch (err) {
      setMessages(prev => [
        ...prev,
        {
          role: 'assistant',
          text: "I encountered a hiccup connecting to the curator engine. Make sure the backend server is running!",
          citations: []
        }
      ]);
    } finally {
      setIsStreaming(false);
    }
  };

  const triggerRssSync = async () => {
    setIsSyncing(true);
    try {
      const res = await fetch(`${API_BASE}/api/sync`, { method: 'POST' });
      const data = await res.json();
      if (data.status === 'success') {
        const moviesRes = await fetch(`${API_BASE}/api/movies?limit=8`);
        const mData = await moviesRes.json();
        if (mData.movies) setRecentWatches(mData.movies);
      }
    } catch (e) {
      console.error("Sync error:", e);
    } finally {
      setIsSyncing(false);
    }
  };

  const handleEscalateSubmit = async (e) => {
    e.preventDefault();
    if (!escalateReason.trim()) return;

    try {
      const res = await fetch(`${API_BASE}/api/escalate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          tenant_id: 'default_tenant',
          user_id: 'guest_user',
          conversation_id: 'conv_pg_session',
          reason: escalateReason,
          transcript_summary: messages.map(m => `${m.role}: ${m.text}`).join('\n').slice(-400)
        })
      });
      const data = await res.json();
      setEscalationTicket(data.ticket_id);
      setIsEscalateOpen(false);
      setMessages(prev => [
        ...prev,
        {
          role: 'assistant',
          text: `🎟️ Your inquiry has been escalated directly to PG (Ticket ID: **${data.ticket_id}**). Pranav will review your conversation history and respond!`,
          citations: []
        }
      ]);
    } catch (err) {
      console.error("Escalation error:", err);
    }
  };

  const quickPrompts = [
    "Recommend a chilling Malayalam horror like Bhoothakaalam",
    "What did PG think of Dune and Denis Villeneuve?",
    "Suggest an atmospheric 90s neo-noir or slasher",
    "Can I talk to PG directly for festival advice?"
  ];

  return (
    <div className="app-viewport">
      {/* Top Navigation Bar */}
      <header className="app-header">
        <div className="brand-section">
          <div className="brand-logo">
            <div className="brand-icon-wrap">
              <Clapperboard size={20} color="#0c0f12" />
            </div>
            <span className="brand-name">
              PG Recommends
              <span className="brand-badge">Letterboxd Diary</span>
            </span>
          </div>
          <div className="curator-pill">
            <span className="status-dot"></span>
            <span>Pranav G (@pranavg)</span>
          </div>
        </div>

        <div className="header-actions">
          <button 
            className="btn-secondary-glass" 
            onClick={triggerRssSync} 
            disabled={isSyncing}
            title="Sync latest reviews from letterboxd.com/pranavg/rss/"
          >
            <RefreshCw size={14} className={isSyncing ? "animate-spin" : ""} />
            <span>{isSyncing ? "Syncing..." : "Sync Letterboxd"}</span>
          </button>

          <button 
            className="btn-escalate" 
            onClick={() => setIsEscalateOpen(true)}
          >
            <MessageSquareShare size={15} />
            <span>Ask PG Directly</span>
          </button>
        </div>
      </header>

      {/* Main 3-Column Glass Layout */}
      <main className="main-layout">
        {/* Left Column: Taste Radar */}
        <aside className="sidebar-left glass-pane">
          <div className="pane-header">
            <span className="pane-title">
              <Compass size={17} color="#40bcf4" />
              Taste Radar
            </span>
          </div>
          <div className="taste-radar-body">
            <div className="taste-metric-card">
              <div className="metric-header">
                <span>Taste Alignment</span>
                <span>Active Profile</span>
              </div>
              <div className="metric-value">94% Fit</div>
              <div className="meter-bar-track">
                <div className="meter-bar-fill" style={{ width: '94%' }}></div>
              </div>
            </div>

            <div>
              <div className="taste-category-title">
                <Film size={13} color="#40bcf4" />
                Affinity Directors
              </div>
              <div className="pills-container">
                {tasteProfile.liked_directors?.map((d, i) => (
                  <span key={i} className="taste-pill director">{d}</span>
                ))}
              </div>
            </div>

            <div>
              <div className="taste-category-title">
                <Heart size={13} color="#00e054" />
                Favorite Genres & Tropes
              </div>
              <div className="pills-container">
                {tasteProfile.liked_genres?.map((g, i) => (
                  <span key={i} className="taste-pill">{g}</span>
                ))}
                {tasteProfile.mood_tags?.map((m, i) => (
                  <span key={i} className="taste-pill">{m}</span>
                ))}
              </div>
            </div>

            <div>
              <div className="taste-category-title">
                <ThumbsDown size={13} color="#ff6b6b" />
                Disliked Elements (Avoided)
              </div>
              <div className="pills-container">
                {tasteProfile.disliked_elements?.map((e, i) => (
                  <span key={i} className="taste-pill disliked">{e}</span>
                ))}
              </div>
            </div>

            {escalationTicket && (
              <div className="taste-metric-card" style={{ borderColor: '#ff8000' }}>
                <div className="metric-header" style={{ color: '#ff8000' }}>
                  <span>Active Human Ticket</span>
                </div>
                <div style={{ fontSize: '0.95rem', fontWeight: 'bold', color: '#fff' }}>
                  {escalationTicket}
                </div>
                <div style={{ fontSize: '0.74rem', color: '#9ab0c2', marginTop: 4 }}>
                  Escalated directly to PG
                </div>
              </div>
            )}
          </div>
        </aside>

        {/* Center Stage: Chat Feed */}
        <section className="chat-container glass-pane">
          <div className="pane-header">
            <span className="pane-title">
              <Sparkles size={17} color="#00e054" />
              Talking Movies with PG
            </span>
            <span style={{ fontSize: '0.75rem', color: '#677b8c' }}>
              Synced with Letterboxd
            </span>
          </div>

          <div className="chat-history">
            {messages.map((msg, idx) => (
              <div key={idx} className={`chat-bubble ${msg.role}`}>
                {msg.role === 'assistant' && (
                  <div className="curator-tagline">
                    <UserCheck size={13} />
                    <span>PG's Take</span>
                  </div>
                )}
                <div className="bubble-content">
                  <div style={{ whiteSpace: 'pre-line' }}>{msg.text}</div>

                  {/* Multi-Source 1: PG's Letterboxd Diary & Acclaimed Cinema */}
                  {msg.citations && msg.citations.length > 0 && (
                    <div style={{ marginTop: '14px', display: 'flex', flexDirection: 'column', gap: '10px' }}>
                      <div className="source-group-title">
                        <span>Film Recommendations & Source Citations</span>
                      </div>
                      {msg.citations.map((cit, cIdx) => (
                        <div key={cIdx} className="movie-rec-card">
                          <div className="rec-card-header">
                            <span className="rec-title">{cit.title} ({cit.year})</span>
                            <span className="rec-rating">
                              {cit.source_type === 'acclaimed_cinema' 
                                ? '★ Acclaimed Cinema' 
                                : (cit.rating ? `★ ${cit.rating} (PG)` : 'PG Logged')}
                            </span>
                          </div>
                          <div className="rec-quote">
                            "{cit.excerpt ? cit.excerpt.replace(/&#039;/g, "'").replace(/&#39;/g, "'").replace(/&quot;/g, '"').replace(/&amp;/g, '&') : ''}"
                          </div>
                          <a 
                            href={cit.letterboxd_url} 
                            target="_blank" 
                            rel="noopener noreferrer" 
                            className="rec-link-btn"
                          >
                            <ExternalLink size={12} />
                            <span>
                              {cit.source_type === 'acclaimed_cinema' 
                                ? `Explore Reception on ${cit.source_portal || 'Critic Portal'}` 
                                : 'View Review on Letterboxd'}
                            </span>
                          </a>
                        </div>
                      ))}
                    </div>
                  )}

                  {/* Multi-Source 2: Reputed Online Critic Reviews & Portals */}
                  {msg.criticCitations && msg.criticCitations.length > 0 && (
                    <div style={{ marginTop: '12px', display: 'flex', flexDirection: 'column', gap: '10px' }}>
                      <div className="source-group-title critic-source-title">
                        <span>Reputed Online Critic Portals & Web Sources</span>
                      </div>
                      {msg.criticCitations.map((crit, crIdx) => (
                        <div key={crIdx} className="critic-rec-card">
                          <div className="rec-card-header">
                            <span className="critic-portal-badge">{crit.portal_name}</span>
                            {crit.critic_name && (
                              <span className="critic-author">{crit.critic_name}</span>
                            )}
                          </div>
                          <div className="rec-quote">
                            "{crit.excerpt ? crit.excerpt.replace(/&#039;/g, "'").replace(/&#39;/g, "'").replace(/&quot;/g, '"').replace(/&amp;/g, '&') : ''}"
                          </div>
                          <a 
                            href={crit.review_url} 
                            target="_blank" 
                            rel="noopener noreferrer" 
                            className="critic-link-btn"
                          >
                            <ExternalLink size={12} />
                            <span>Read on {crit.portal_name}</span>
                          </a>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>

          {/* Bottom Input Area */}
          <div className="chat-input-bar">
            <form 
              onSubmit={(e) => { e.preventDefault(); handleSend(); }} 
              className="input-container"
            >
              <input
                type="text"
                className="chat-input"
                placeholder="Ask me for a movie recommendation, compare films, or talk about what you love..."
                value={input}
                onChange={(e) => setInput(e.target.value)}
                disabled={isStreaming}
              />
              <button 
                type="submit" 
                className="btn-send" 
                disabled={!input.trim() || isStreaming}
              >
                <Send size={16} />
              </button>
            </form>
          </div>
        </section>

        {/* Right Sidebar: Recent Letterboxd Watches & Starter Chips */}
        <aside className="sidebar-right glass-pane">
          <div className="pane-header">
            <span className="pane-title">
              <Film size={17} color="#00e054" />
              PG's Letterboxd Diary
            </span>
          </div>

          <div className="recent-watches-feed">
            <div className="taste-category-title" style={{ marginTop: 0 }}>
              Recent Watches & High Ratings
            </div>
            {recentWatches.map((movie, i) => (
              <div 
                key={i} 
                className="movie-mini-card"
                onClick={() => handleSend(`Tell me your full thoughts on ${movie.title} (${movie.year})`)}
              >
                <div className="mini-card-title">
                  <span>{movie.title}</span>
                  <span className="mini-card-rating">
                    {movie.rating ? `★ ${movie.rating}` : ''}
                  </span>
                </div>
                <div className="mini-card-snippet">
                  {movie.review_text || `Logged on ${movie.watched_date || 'Letterboxd'}`}
                </div>
              </div>
            ))}

            <div className="quick-prompts-section">
              <div className="taste-category-title">
                Conversation Starters
              </div>
              {quickPrompts.map((prompt, i) => (
                <button
                  key={i}
                  className="quick-chip"
                  onClick={() => handleSend(prompt)}
                  disabled={isStreaming}
                >
                  "{prompt}"
                </button>
              ))}
            </div>
          </div>
        </aside>
      </main>

      {/* Modal: Human Escalation */}
      {isEscalateOpen && (
        <div className="modal-backdrop" onClick={() => setIsEscalateOpen(false)}>
          <div className="modal-dialog" onClick={(e) => e.stopPropagation()}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div className="modal-title">
                <AlertCircle size={20} />
                <span>Ask PG Directly</span>
              </div>
              <button 
                style={{ background: 'none', border: 'none', color: '#9ab0c2', cursor: 'pointer' }}
                onClick={() => setIsEscalateOpen(false)}
              >
                <X size={18} />
              </button>
            </div>

            <p style={{ fontSize: '0.86rem', color: '#9ab0c2', lineHeight: 1.5 }}>
              Need a bespoke film festival itinerary, personalized curation, or have a specific question for the real human curator? Submit your inquiry here and it will be dispatched to PG's queue.
            </p>

            <form onSubmit={handleEscalateSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
              <div className="form-field">
                <label className="form-label">What would you like to ask or request from PG?</label>
                <textarea
                  className="form-textarea"
                  rows={4}
                  placeholder="e.g., I'm attending MIFF next week and need a 5-movie watchlist based on your favorite Asian cinema picks..."
                  value={escalateReason}
                  onChange={(e) => setEscalateReason(e.target.value)}
                  required
                />
              </div>

              <div className="modal-actions">
                <button 
                  type="button" 
                  className="btn-secondary-glass" 
                  onClick={() => setIsEscalateOpen(false)}
                >
                  Cancel
                </button>
                <button 
                  type="submit" 
                  className="btn-escalate"
                >
                  Submit Ticket to PG
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
