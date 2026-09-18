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
  X,
  LogIn,
  LogOut,
  User,
  Zap,
  CheckCircle
} from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_BASE ?? (import.meta.env.PROD ? '' : 'http://localhost:8000');

function renderInlineMarkdown(text) {
  if (!text) return null;
  const regex = /(\*\*.*?\*\*|\*[^*]+?\*|★\s*\d+(?:\.\d+)?)/g;
  const parts = [];
  let lastIdx = 0;
  let match;

  while ((match = regex.exec(text)) !== null) {
    if (match.index > lastIdx) {
      parts.push(text.substring(lastIdx, match.index));
    }
    const token = match[0];
    if (token.startsWith('**') && token.endsWith('**')) {
      parts.push(
        <strong key={match.index} className="msg-strong">
          {token.slice(2, -2)}
        </strong>
      );
    } else if (token.startsWith('★')) {
      parts.push(
        <span key={match.index} className="msg-star-badge">
          {token}
        </span>
      );
    } else if (token.startsWith('*') && token.endsWith('*')) {
      parts.push(
        <em key={match.index} className="msg-em">
          {token.slice(1, -1)}
        </em>
      );
    } else {
      parts.push(token);
    }
    lastIdx = regex.lastIndex;
  }
  if (lastIdx < text.length) {
    parts.push(text.substring(lastIdx));
  }
  return parts.length > 0 ? parts : text;
}

function FormattedMessage({ text, isStreaming }) {
  if (!text) return null;

  const blocks = text.split(/\n\s*\n/);

  return (
    <div className="msg-formatted-content">
      {blocks.map((block, bIdx) => {
        const trimmed = block.trim();
        if (!trimmed) return null;

        const lines = trimmed.split('\n');
        const firstLine = lines[0].trim();
        const isMovieCard = firstLine.startsWith('🎬') || firstLine.startsWith('🌍');

        if (isMovieCard) {
          return (
            <div key={bIdx} className="msg-rec-card">
              {lines.map((line, lIdx) => {
                const lineTrimmed = line.trim();
                if (!lineTrimmed) return null;
                if (lineTrimmed.startsWith('🎬') || lineTrimmed.startsWith('🌍')) {
                  return (
                    <div key={lIdx} className="msg-rec-header">
                      {renderInlineMarkdown(lineTrimmed)}
                    </div>
                  );
                } else if (lineTrimmed.startsWith('>')) {
                  const quoteContent = lineTrimmed.replace(/^>\s*/, '');
                  return (
                    <blockquote key={lIdx} className="msg-rec-quote">
                      {renderInlineMarkdown(quoteContent)}
                    </blockquote>
                  );
                } else {
                  return (
                    <div key={lIdx} className="msg-rec-critic">
                      {renderInlineMarkdown(lineTrimmed)}
                    </div>
                  );
                }
              })}
            </div>
          );
        }

        if (lines.every((l) => l.trim().startsWith('>'))) {
          const quoteText = lines.map((l) => l.trim().replace(/^>\s*/, '')).join(' ');
          return (
            <blockquote key={bIdx} className="msg-rec-quote">
              {renderInlineMarkdown(quoteText)}
            </blockquote>
          );
        }

        return (
          <p key={bIdx} className="msg-paragraph">
            {lines.map((line, lIdx) => (
              <span key={lIdx}>
                {renderInlineMarkdown(line)}
                {lIdx < lines.length - 1 && <br />}
              </span>
            ))}
          </p>
        );
      })}
      {isStreaming && <span className="streaming-cursor"></span>}
    </div>
  );
}

export default function App() {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      text: "Hey! Pranav here. Tell me what kind of mood you're in, what you've watched recently, or what genres you want to explore, and I'll dig into my Letterboxd diary to hook you up with something genuinely great. What are you feeling today?",
      citations: [],
      criticCitations: []
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
  const [mobileTab, setMobileTab] = useState('chat'); // 'chat' | 'taste' | 'diary'

  // User Auth & Taste Matching State
  const [token, setToken] = useState(() => localStorage.getItem('pg_auth_token') || '');
  const [currentUser, setCurrentUser] = useState(null);
  const [isAuthOpen, setIsAuthOpen] = useState(false);
  const [authTab, setAuthTab] = useState('signin'); // 'signin' | 'register'
  const [authEmail, setAuthEmail] = useState('');
  const [authPassword, setAuthPassword] = useState('');
  const [authHandle, setAuthHandle] = useState('');
  const [authError, setAuthError] = useState('');
  const [authLoading, setAuthLoading] = useState(false);
  const [isSyncingUser, setIsSyncingUser] = useState(false);

  const messagesEndRef = useRef(null);

  // Auto-scroll to bottom of chat
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Check existing auth token
  useEffect(() => {
    if (token) {
      fetch(`${API_BASE}/api/auth/me`, {
        headers: { 'Authorization': `Bearer ${token}` }
      })
        .then(res => res.json())
        .then(data => {
          if (data.authenticated && data.user) {
            setCurrentUser(data.user);
          } else {
            localStorage.removeItem('pg_auth_token');
            setToken('');
            setCurrentUser(null);
          }
        })
        .catch(() => {
          // Keep guest mode if offline
        });
    }
  }, [token]);

  // Load initial catalog & taste profile
  useEffect(() => {
    fetch(`${API_BASE}/api/movies?limit=8`)
      .then(res => res.json())
      .then(data => {
        if (data.movies) setRecentWatches(data.movies);
      })
      .catch(err => console.log("Backend offline or loading:", err));

    const activeUserId = currentUser?.id || 'guest_user';
    fetch(`${API_BASE}/api/taste-profile?tenant_id=default_tenant&user_id=${activeUserId}`)
      .then(res => res.json())
      .then(data => {
        if (data.liked_genres?.length > 0 || data.liked_directors?.length > 0) {
          setTasteProfile(data);
        }
      })
      .catch(() => {});
  }, [currentUser]);

  const handleAuthSubmit = async (e) => {
    e.preventDefault();
    setAuthError('');
    setAuthLoading(true);

    const endpoint = authTab === 'signin' ? '/api/auth/login' : '/api/auth/register';
    const payload = authTab === 'signin' 
      ? { email: authEmail, password: authPassword }
      : { email: authEmail, password: authPassword, letterboxd_handle: authHandle.trim() || null };

    try {
      const res = await fetch(`${API_BASE}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || 'Authentication failed. Please check your credentials.');
      }

      localStorage.setItem('pg_auth_token', data.access_token);
      setToken(data.access_token);
      setCurrentUser(data.user);
      setIsAuthOpen(false);
      setAuthPassword('');
      setAuthError('');

      // Add a friendly greeting message
      const welcomeMsg = authTab === 'register' && data.user.letterboxd_handle
        ? `Welcome to PG Recommends, @${data.user.letterboxd_handle}! We analyzed your Letterboxd diary: your Taste Match with PG is **${data.user.taste_match_pct}%**! Ask me anything.`
        : `Welcome back! Your taste profile is synced and ready. What are we watching today?`;

      setMessages(prev => [
        ...prev,
        { role: 'assistant', text: welcomeMsg, citations: [], criticCitations: [] }
      ]);
    } catch (err) {
      setAuthError(err.message);
    } finally {
      setAuthLoading(false);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('pg_auth_token');
    setToken('');
    setCurrentUser(null);
  };

  const handleSyncUserLetterboxd = async () => {
    if (!currentUser) {
      setIsAuthOpen(true);
      return;
    }
    const handle = prompt("Enter your public Letterboxd username:", currentUser.letterboxd_handle || "");
    if (!handle || !handle.trim()) return;

    setIsSyncingUser(true);
    try {
      const res = await fetch(`${API_BASE}/api/user/sync-letterboxd`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({ username: handle.trim() })
      });
      const data = await res.json();
      if (res.ok && data.taste_match_pct) {
        setCurrentUser(prev => ({
          ...prev,
          letterboxd_handle: data.username,
          taste_match_pct: data.taste_match_pct
        }));
        if (data.taste_profile) {
          setTasteProfile(data.taste_profile);
        }
        alert(`Successfully synced @${data.username}! Your updated Taste Match with PG is ${data.taste_match_pct}%.`);
      } else {
        alert(data.detail || "Could not sync Letterboxd feed.");
      }
    } catch (e) {
      alert("Error syncing Letterboxd feed: " + e.message);
    } finally {
      setIsSyncingUser(false);
    }
  };

  const handleSend = async (messageText = input) => {
    const trimmed = messageText.trim();
    if (!trimmed || isStreaming) return;

    setInput('');
    const userMsg = { role: 'user', text: trimmed };
    setMessages(prev => [...prev, userMsg]);
    setIsStreaming(true);

    setMessages(prev => [
      ...prev,
      { role: 'assistant', text: '', citations: [], criticCitations: [], isStreaming: true, fromCache: false }
    ]);

    try {
      const headers = { 'Content-Type': 'application/json' };
      if (token) {
        headers['Authorization'] = `Bearer ${token}`;
      }

      const response = await fetch(`${API_BASE}/api/chat/stream`, {
        method: 'POST',
        headers: headers,
        body: JSON.stringify({
          tenant_id: 'default_tenant',
          user_id: currentUser?.id || 'guest_user',
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
      let fromCache = false;

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

              if (eventData.event === 'cache_hit') {
                fromCache = true;
              } else if (eventData.event === 'token') {
                streamedText += eventData.data;
                setMessages(prev => {
                  const updated = [...prev];
                  updated[updated.length - 1] = {
                    role: 'assistant',
                    text: streamedText,
                    citations: citations,
                    criticCitations: criticCitations,
                    isStreaming: true,
                    fromCache: fromCache
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
                    isStreaming: true,
                    fromCache: fromCache
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
                    isStreaming: true,
                    fromCache: fromCache
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
          criticCitations: criticCitations,
          fromCache: fromCache
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
          user_id: currentUser?.id || 'guest_user',
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

  const tastePercentage = currentUser?.taste_match_pct || 72.0;

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
          {currentUser ? (
            <div className="user-auth-badge">
              <User size={14} color="#40bcf4" />
              <span>{currentUser.letterboxd_handle ? `@${currentUser.letterboxd_handle}` : currentUser.email.split('@')[0]}</span>
              <span className="user-taste-badge">{currentUser.taste_match_pct}% Match</span>
              <button 
                type="button" 
                className="btn-auth-logout" 
                onClick={handleLogout}
                title="Log out of account"
              >
                Sign out
              </button>
            </div>
          ) : (
            <button 
              type="button" 
              className="btn-auth-signin"
              onClick={() => { setIsAuthOpen(true); setAuthTab('signin'); }}
            >
              <LogIn size={14} />
              <span>Sign In</span>
            </button>
          )}

          <button 
            className="btn-secondary-glass" 
            onClick={triggerRssSync} 
            disabled={isSyncing}
            title="Sync latest reviews from letterboxd.com/pranavg/rss/"
          >
            <RefreshCw size={14} className={isSyncing ? "animate-spin" : ""} />
            <span className="btn-label-desktop">{isSyncing ? "Syncing..." : "Sync Letterboxd"}</span>
            <span className="btn-label-mobile">{isSyncing ? "Sync..." : "Sync"}</span>
          </button>

          <button 
            className="btn-escalate" 
            onClick={() => setIsEscalateOpen(true)}
          >
            <MessageSquareShare size={15} />
            <span className="btn-label-desktop">Ask PG Directly</span>
            <span className="btn-label-mobile">Ask PG</span>
          </button>
        </div>
      </header>

      {/* Mobile Tab Segmented Switcher (visible on mobile / small screens) */}
      <nav className="mobile-nav-bar" aria-label="Mobile View Navigation">
        <button 
          type="button"
          className={`mobile-tab-btn ${mobileTab === 'chat' ? 'active' : ''}`}
          onClick={() => setMobileTab('chat')}
        >
          <Sparkles size={14} />
          <span>Chat</span>
        </button>
        <button 
          type="button"
          className={`mobile-tab-btn ${mobileTab === 'taste' ? 'active' : ''}`}
          onClick={() => setMobileTab('taste')}
        >
          <Compass size={14} />
          <span>Taste Radar</span>
        </button>
        <button 
          type="button"
          className={`mobile-tab-btn ${mobileTab === 'diary' ? 'active' : ''}`}
          onClick={() => setMobileTab('diary')}
        >
          <Film size={14} />
          <span>PG's Diary</span>
        </button>
      </nav>

      {/* Main 3-Column Glass Layout */}
      <main className="main-layout">
        {/* Left Column: Taste Radar */}
        <aside className={`sidebar-left glass-pane ${mobileTab === 'taste' ? 'mobile-visible' : ''}`}>
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
                <span>{currentUser ? (currentUser.letterboxd_handle ? `@${currentUser.letterboxd_handle}` : 'Your Account') : 'Guest Profile'}</span>
              </div>
              <div className="metric-value">{tastePercentage}% Match</div>
              <div className="meter-bar-track">
                <div className="meter-bar-fill" style={{ width: `${tastePercentage}%` }}></div>
              </div>
              <div style={{ marginTop: 8, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: '0.74rem', color: '#9ab0c2' }}>
                  {currentUser ? 'Objective compatibility with PG' : 'Sign in to match your Letterboxd'}
                </span>
                <button 
                  type="button" 
                  onClick={handleSyncUserLetterboxd}
                  disabled={isSyncingUser}
                  style={{
                    background: 'none',
                    border: 'none',
                    color: '#40bcf4',
                    fontSize: '0.74rem',
                    cursor: 'pointer',
                    fontWeight: 600,
                    textDecoration: 'underline'
                  }}
                >
                  {isSyncingUser ? 'Syncing...' : (currentUser?.letterboxd_handle ? 'Re-sync RSS' : 'Connect Letterboxd')}
                </button>
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
        <section className={`chat-container glass-pane ${mobileTab === 'chat' ? 'mobile-visible' : ''}`}>
          <div className="pane-header">
            <span className="pane-title">
              <Sparkles size={17} color="#00e054" />
              Talking Movies with PG
            </span>
            <span style={{ fontSize: '0.75rem', color: '#677b8c' }}>
              Synced with Letterboxd & Acclaimed Cinema
            </span>
          </div>

          <div className="chat-history">
            {messages.map((msg, idx) => (
              <div 
                key={idx} 
                className={`chat-bubble ${msg.role} ${msg.isStreaming && !msg.text ? 'is-thinking' : ''}`}
              >
                {msg.role === 'assistant' && (
                  <div className="curator-tagline">
                    <UserCheck size={13} />
                    <span>PG's Take</span>
                    {msg.fromCache && (
                      <span className="cache-indicator-badge">
                        <Zap size={10} />
                        Semantic Cache (&lt;5ms)
                      </span>
                    )}
                    {msg.isStreaming && (
                      <span className="live-thinking-indicator">
                        <span className="live-pulse"></span>
                        {msg.text ? 'Streaming' : 'Consulting diary...'}
                      </span>
                    )}
                  </div>
                )}
                <div className="bubble-content">
                  {msg.isStreaming && !msg.text ? (
                    <div className="thinking-container">
                      <div className="thinking-dots">
                        <span className="thinking-dot dot-1"></span>
                        <span className="thinking-dot dot-2"></span>
                        <span className="thinking-dot dot-3"></span>
                      </div>
                      <span className="thinking-label">
                        Consulting Letterboxd diary, ratings & critic reviews...
                      </span>
                    </div>
                  ) : (
                    <FormattedMessage text={msg.text} isStreaming={msg.isStreaming} />
                  )}

                  {/* Film Citations with High-Res Posters */}
                  {msg.citations && msg.citations.length > 0 && (
                    <div style={{ marginTop: '14px', display: 'flex', flexDirection: 'column', gap: '10px' }}>
                      <div className="source-group-title">
                        <span>Film Recommendations & Source Citations</span>
                      </div>
                      {msg.citations.map((cit, cIdx) => (
                        <div key={cIdx} className="movie-rec-card">
                          {cit.poster_url && (
                            <div className="rec-poster-wrap">
                              <img 
                                src={cit.poster_url} 
                                alt={cit.title} 
                                className="rec-poster-img"
                                loading="lazy"
                                onError={(e) => { e.target.style.display = 'none'; }}
                              />
                            </div>
                          )}
                          <div className="rec-card-body">
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
        <aside className={`sidebar-right glass-pane ${mobileTab === 'diary' ? 'mobile-visible' : ''}`}>
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
                onClick={() => {
                  handleSend(`Tell me your full thoughts on ${movie.title} (${movie.year})`);
                  setMobileTab('chat');
                }}
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
                  onClick={() => {
                    handleSend(prompt);
                    setMobileTab('chat');
                  }}
                  disabled={isStreaming}
                >
                  "{prompt}"
                </button>
              ))}
            </div>
          </div>
        </aside>
      </main>

      {/* Modal: User Authentication (Login / Register) */}
      {isAuthOpen && (
        <div className="modal-backdrop" onClick={() => setIsAuthOpen(false)}>
          <div className="modal-dialog" onClick={(e) => e.stopPropagation()}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
              <div className="modal-title">
                <User size={20} color="#00e054" />
                <span>{authTab === 'signin' ? 'Sign In to PG Recommends' : 'Create Cinephile Account'}</span>
              </div>
              <button 
                style={{ background: 'none', border: 'none', color: '#9ab0c2', cursor: 'pointer' }}
                onClick={() => setIsAuthOpen(false)}
              >
                <X size={18} />
              </button>
            </div>

            <div className="auth-tabs">
              <button 
                type="button" 
                className={`auth-tab-btn ${authTab === 'signin' ? 'active' : ''}`}
                onClick={() => { setAuthTab('signin'); setAuthError(''); }}
              >
                Sign In
              </button>
              <button 
                type="button" 
                className={`auth-tab-btn ${authTab === 'register' ? 'active' : ''}`}
                onClick={() => { setAuthTab('register'); setAuthError(''); }}
              >
                Create Account
              </button>
            </div>

            {authError && (
              <div style={{
                background: 'rgba(255, 107, 107, 0.15)',
                border: '1px solid rgba(255, 107, 107, 0.35)',
                color: '#ff6b6b',
                padding: '8px 12px',
                borderRadius: '8px',
                fontSize: '0.82rem',
                marginBottom: 14
              }}>
                {authError}
              </div>
            )}

            <form onSubmit={handleAuthSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              <div className="form-field">
                <label className="form-label">Email Address</label>
                <input 
                  type="email" 
                  className="chat-input"
                  style={{ borderRadius: '8px', padding: '10px 14px' }}
                  placeholder="cinephile@example.com"
                  value={authEmail}
                  onChange={(e) => setAuthEmail(e.target.value)}
                  required
                />
              </div>

              <div className="form-field">
                <label className="form-label">Password</label>
                <input 
                  type="password" 
                  className="chat-input"
                  style={{ borderRadius: '8px', padding: '10px 14px' }}
                  placeholder="••••••••"
                  value={authPassword}
                  onChange={(e) => setAuthPassword(e.target.value)}
                  required
                />
              </div>

              {authTab === 'register' && (
                <div className="form-field">
                  <label className="form-label">Letterboxd Username (Optional)</label>
                  <input 
                    type="text" 
                    className="chat-input"
                    style={{ borderRadius: '8px', padding: '10px 14px' }}
                    placeholder="e.g. pranavg or your username"
                    value={authHandle}
                    onChange={(e) => setAuthHandle(e.target.value)}
                  />
                  <span style={{ fontSize: '0.73rem', color: '#677b8c', marginTop: 4 }}>
                    Syncs your public diary & automatically calculates your Taste Match % against PG.
                  </span>
                </div>
              )}

              <div className="modal-actions" style={{ marginTop: 10 }}>
                <button 
                  type="button" 
                  className="btn-secondary-glass" 
                  onClick={() => setIsAuthOpen(false)}
                >
                  Cancel
                </button>
                <button 
                  type="submit" 
                  className="btn-escalate"
                  disabled={authLoading}
                >
                  {authLoading ? 'Processing...' : (authTab === 'signin' ? 'Sign In' : 'Create Account')}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

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
