import React, { useState, useEffect } from 'react';
import { createPortal } from 'react-dom';

// Simple inline icon components
const IconX = ({ className }) => (
  <svg className={className} width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <line x1="18" y1="6" x2="6" y2="18"></line>
    <line x1="6" y1="6" x2="18" y2="18"></line>
  </svg>
);
const IconRefresh = ({ className }) => (
  <svg className={className} width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M23 4v6h-6"></path>
    <path d="M1 20v-6h6"></path>
    <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"></path>
  </svg>
);
const IconClock = ({ className }) => (
  <svg className={className} width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <circle cx="12" cy="12" r="10"></circle>
    <polyline points="12 6 12 12 16 14"></polyline>
  </svg>
);
const IconStar = ({ className }) => (
  <svg className={className} width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon>
  </svg>
);
const IconChevronDown = ({ className }) => (
  <svg className={className} width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <polyline points="6 9 12 15 18 9"></polyline>
  </svg>
);
const IconAlertCircle = ({ className }) => (
  <svg className={className} width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <circle cx="12" cy="12" r="10"></circle>
    <line x1="12" y1="8" x2="12" y2="12"></line>
    <line x1="12" y1="16" x2="12.01" y2="16"></line>
  </svg>
);

/**
 * Get color class based on rating value (1-10)
 */
function getRatingColor(rating) {
  if (rating >= 8) return 'text-emerald-400';
  if (rating >= 6) return 'text-blue-400';
  if (rating >= 4) return 'text-yellow-400';
  if (rating >= 2) return 'text-orange-400';
  return 'text-red-400';
}

function getRatingBgColor(rating) {
  if (rating >= 8) return 'bg-emerald-500';
  if (rating >= 6) return 'bg-blue-500';
  if (rating >= 4) return 'bg-yellow-500';
  if (rating >= 2) return 'bg-orange-500';
  return 'bg-red-500';
}

/**
 * Rating display component with progress bar
 */
function RatingDisplay({ label, rating, feedback }) {
  const percentage = (rating / 10) * 100;
  
  return (
    <div className="bg-gray-800/50 rounded-lg p-4">
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm font-medium text-gray-300">{label}</span>
        <span className={`text-2xl font-bold ${getRatingColor(rating)}`}>
          {rating > 0 ? rating : '—'}/10
        </span>
      </div>
      
      {/* Progress bar */}
      <div className="h-2 bg-gray-700 rounded-full mb-3 overflow-hidden">
        <div 
          className={`h-full rounded-full transition-all duration-500 ${getRatingBgColor(rating)}`}
          style={{ width: `${percentage}%` }}
        />
      </div>
      
      {/* Feedback text */}
      {feedback && (
        <p className="text-sm text-gray-400 leading-relaxed">{feedback}</p>
      )}
    </div>
  );
}

/**
 * Modal for displaying paper critiques from the validator model.
 * 
 * Props:
 * - isOpen: boolean - whether the modal is visible
 * - onClose: function - callback when modal is closed
 * - paperType: 'autonomous_paper' | 'final_answer' | 'compiler_paper'
 * - paperId: string - the paper ID (for autonomous papers) or answer ID (for final answers)
 * - paperTitle: string - title of the paper being critiqued
 * - onGenerateCritique: async function(customPrompt) - callback to generate a new critique
 * - onGetCritiques: async function() - callback to get critique history
 */
export default function PaperCritiqueModal({
  isOpen,
  onClose,
  paperType,
  paperId,
  paperTitle,
  onGenerateCritique,
  onGetCritiques,
}) {
  const [loading, setLoading] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState(null);
  const [critiques, setCritiques] = useState([]);
  const [selectedCritique, setSelectedCritique] = useState(null);
  const [historyOpen, setHistoryOpen] = useState(false);

  // Load critiques when modal opens
  useEffect(() => {
    if (isOpen && onGetCritiques) {
      loadCritiques();
    }
  }, [isOpen, paperId]);

  const loadCritiques = async () => {
    try {
      setLoading(true);
      setError(null);
      const result = await onGetCritiques();
      const critiqueList = result.critiques || [];
      setCritiques(critiqueList);
      
      // Select the most recent critique if available
      if (critiqueList.length > 0) {
        setSelectedCritique(critiqueList[0]);
      } else {
        setSelectedCritique(null);
      }
    } catch (err) {
      console.error('Failed to load critiques:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleGenerateCritique = async () => {
    try {
      setGenerating(true);
      setError(null);
      
      // Get custom prompt from localStorage if available
      const storageKey = paperType === 'compiler_paper' 
        ? 'compiler_critique_custom_prompt'
        : 'autonomous_critique_custom_prompt';
      const customPrompt = localStorage.getItem(storageKey);
      
      // Get validator config from localStorage (allows critiques without starting research)
      let validatorConfig = null;
      try {
        const configKey = paperType === 'compiler_paper' ? 'compiler_settings' : 'autonomousConfig';
        const configStr = localStorage.getItem(configKey);
        if (configStr) {
          const config = JSON.parse(configStr);
          // Extract validator config fields
          validatorConfig = {
            validator_model: config.validator_model,
            validator_context_window: config.validator_context_window,
            validator_max_tokens: config.validator_max_tokens,
            validator_provider: config.validator_provider,
            validator_openrouter_provider: config.validator_openrouter_provider,
          };
        }
      } catch (e) {
        console.warn('Could not read validator config from localStorage:', e);
      }
      
      const result = await onGenerateCritique(customPrompt, validatorConfig);
      
      // Reload critiques to get the updated list
      await loadCritiques();
    } catch (err) {
      console.error('Failed to generate critique:', err);
      setError(err.message);
    } finally {
      setGenerating(false);
    }
  };

  const formatDate = (dateStr) => {
    if (!dateStr) return 'Unknown date';
    const date = new Date(dateStr);
    return date.toLocaleString();
  };

  if (!isOpen) return null;

  // Use createPortal to render at document.body level, bypassing any parent stacking contexts
  const modalContent = (
    <div 
      className="critique-modal-overlay"
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundColor: 'rgba(0, 0, 0, 0.7)',
        backdropFilter: 'blur(4px)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 999999, // Very high z-index to ensure visibility
        padding: '20px',
      }}
      onClick={(e) => {
        // Close when clicking the backdrop
        if (e.target === e.currentTarget) {
          onClose();
        }
      }}
    >
      <div 
        className="critique-modal-content"
        style={{
          backgroundColor: '#1a1a2e',
          borderRadius: '12px',
          width: '100%',
          maxWidth: '1200px',
          height: '85vh',
          display: 'flex',
          flexDirection: 'column',
          boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.8), 0 0 0 1px rgba(147, 51, 234, 0.3)',
          border: '1px solid rgba(147, 51, 234, 0.4)',
          overflow: 'hidden',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header - Compact */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '12px 16px',
          borderBottom: '1px solid rgba(75, 85, 99, 0.5)',
          backgroundColor: 'rgba(147, 51, 234, 0.1)',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <div style={{
              padding: '6px',
              backgroundColor: 'rgba(147, 51, 234, 0.2)',
              borderRadius: '8px',
            }}>
              <IconStar className="w-4 h-4 text-purple-400" />
            </div>
            <div>
              <h2 style={{ 
                fontSize: '14px', 
                fontWeight: '600', 
                color: '#f3f4f6',
                margin: 0,
              }}>Validator Critique</h2>
              <p style={{ 
                fontSize: '11px', 
                color: '#9ca3af',
                margin: 0,
                maxWidth: '300px',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
              }} title={paperTitle}>
                {paperTitle || 'Paper'}
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            style={{
              color: '#9ca3af',
              padding: '6px',
              backgroundColor: 'transparent',
              border: 'none',
              borderRadius: '6px',
              cursor: 'pointer',
              transition: 'all 0.2s',
            }}
            onMouseEnter={(e) => {
              e.target.style.backgroundColor = 'rgba(75, 85, 99, 0.5)';
              e.target.style.color = '#f3f4f6';
            }}
            onMouseLeave={(e) => {
              e.target.style.backgroundColor = 'transparent';
              e.target.style.color = '#9ca3af';
            }}
          >
            <IconX className="w-4 h-4" />
          </button>
        </div>

        {/* Content - Scrollable */}
        <div style={{
          flex: 1,
          overflowY: 'auto',
          padding: '16px',
          minHeight: '500px',
        }}>
          {loading ? (
            <div style={{ 
              display: 'flex', 
              alignItems: 'center', 
              justifyContent: 'center',
              height: '150px',
            }}>
              <div style={{ textAlign: 'center' }}>
                <div style={{
                  width: '28px',
                  height: '28px',
                  border: '2px solid #a855f7',
                  borderTopColor: 'transparent',
                  borderRadius: '50%',
                  margin: '0 auto 10px',
                  animation: 'spin 1s linear infinite',
                }}></div>
                <p style={{ color: '#9ca3af', fontSize: '13px' }}>Loading critiques...</p>
              </div>
            </div>
          ) : error ? (
            <div style={{
              backgroundColor: 'rgba(127, 29, 29, 0.2)',
              border: '1px solid rgba(239, 68, 68, 0.3)',
              borderRadius: '8px',
              padding: '12px',
            }}>
              <div style={{ display: 'flex', alignItems: 'flex-start', gap: '10px' }}>
                <IconAlertCircle className="w-4 h-4 text-red-400" style={{ flexShrink: 0, marginTop: '2px' }} />
                <div>
                  <h4 style={{ color: '#f87171', fontWeight: '500', marginBottom: '4px', fontSize: '13px' }}>Error</h4>
                  <p style={{ fontSize: '12px', color: 'rgba(252, 165, 165, 0.8)' }}>{error}</p>
                </div>
              </div>
            </div>
          ) : selectedCritique ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', height: '100%' }}>
              {/* Critic Identity - Compact */}
              <div style={{
                background: 'linear-gradient(to right, rgba(88, 28, 135, 0.3), rgba(30, 58, 138, 0.3))',
                borderRadius: '8px',
                padding: '10px 12px',
                border: '1px solid rgba(147, 51, 234, 0.2)',
              }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <div>
                    <p style={{ fontSize: '10px', color: '#9ca3af', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '2px' }}>Critique by</p>
                    <p style={{ fontSize: '14px', fontWeight: '600', color: '#f3f4f6' }}>{selectedCritique.model_id}</p>
                    {selectedCritique.host_provider && (
                      <p style={{ fontSize: '11px', color: '#c4b5fd' }}>via {selectedCritique.host_provider}</p>
                    )}
                  </div>
                  <div style={{ textAlign: 'right' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '4px', color: '#9ca3af', fontSize: '11px' }}>
                      <IconClock className="w-3 h-3" />
                      {formatDate(selectedCritique.date)}
                    </div>
                  </div>
                </div>
              </div>

              {/* Ratings - Compact Grid */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '8px' }}>
                <CompactRating label="Novelty" rating={selectedCritique.novelty_rating} feedback={selectedCritique.novelty_feedback} />
                <CompactRating label="Correctness" rating={selectedCritique.correctness_rating} feedback={selectedCritique.correctness_feedback} />
                <CompactRating label="Impact" rating={selectedCritique.impact_rating} feedback={selectedCritique.impact_feedback} />
              </div>

              {/* Full Critique - Expanded to fill space */}
              {selectedCritique.full_critique && (
                <div style={{
                  backgroundColor: 'rgba(31, 41, 55, 0.5)',
                  borderRadius: '8px',
                  padding: '10px 12px',
                  flex: 1,
                  display: 'flex',
                  flexDirection: 'column',
                  minHeight: '200px',
                }}>
                  <h3 style={{ fontSize: '11px', fontWeight: '500', color: '#d1d5db', marginBottom: '8px' }}>Full Critique</h3>
                  <p style={{ 
                    color: '#d1d5db', 
                    lineHeight: '1.5', 
                    whiteSpace: 'pre-wrap',
                    fontSize: '12px',
                    flex: 1,
                    overflowY: 'auto',
                  }}>
                    {selectedCritique.full_critique}
                  </p>
                </div>
              )}

              {/* History - Compact */}
              {critiques.length > 1 && (
                <div style={{
                  border: '1px solid rgba(75, 85, 99, 0.5)',
                  borderRadius: '8px',
                  overflow: 'hidden',
                }}>
                  <button
                    onClick={() => setHistoryOpen(!historyOpen)}
                    style={{
                      width: '100%',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      padding: '8px 12px',
                      backgroundColor: 'rgba(31, 41, 55, 0.3)',
                      border: 'none',
                      cursor: 'pointer',
                      transition: 'background-color 0.2s',
                    }}
                    onMouseEnter={(e) => e.target.style.backgroundColor = 'rgba(31, 41, 55, 0.5)'}
                    onMouseLeave={(e) => e.target.style.backgroundColor = 'rgba(31, 41, 55, 0.3)'}
                  >
                    <span style={{ fontSize: '12px', fontWeight: '500', color: '#d1d5db' }}>
                      History ({critiques.length})
                    </span>
                    <IconChevronDown 
                      className="w-3 h-3 text-gray-400"
                      style={{ 
                        transition: 'transform 0.2s',
                        transform: historyOpen ? 'rotate(180deg)' : 'rotate(0deg)',
                      }}
                    />
                  </button>
                  
                  {historyOpen && (
                    <div style={{ 
                      borderTop: '1px solid rgba(75, 85, 99, 0.5)',
                      maxHeight: '120px',
                      overflowY: 'auto',
                    }}>
                      {critiques.map((critique, idx) => (
                        <button
                          key={critique.critique_id || idx}
                          onClick={() => {
                            setSelectedCritique(critique);
                            setHistoryOpen(false);
                          }}
                          style={{
                            width: '100%',
                            textAlign: 'left',
                            padding: '8px 12px',
                            backgroundColor: selectedCritique?.critique_id === critique.critique_id ? 'rgba(88, 28, 135, 0.2)' : 'transparent',
                            border: 'none',
                            borderBottom: idx < critiques.length - 1 ? '1px solid rgba(75, 85, 99, 0.3)' : 'none',
                            cursor: 'pointer',
                            transition: 'background-color 0.2s',
                          }}
                          onMouseEnter={(e) => {
                            if (selectedCritique?.critique_id !== critique.critique_id) {
                              e.target.style.backgroundColor = 'rgba(31, 41, 55, 0.5)';
                            }
                          }}
                          onMouseLeave={(e) => {
                            if (selectedCritique?.critique_id !== critique.critique_id) {
                              e.target.style.backgroundColor = 'transparent';
                            }
                          }}
                        >
                          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                            <span style={{ fontSize: '11px', color: '#d1d5db' }}>{critique.model_id}</span>
                            <span style={{ fontSize: '10px', color: '#6b7280' }}>{formatDate(critique.date)}</span>
                          </div>
                          <div style={{ display: 'flex', gap: '8px', marginTop: '2px', fontSize: '10px' }}>
                            <span className={getRatingColor(critique.novelty_rating)}>N: {critique.novelty_rating}</span>
                            <span className={getRatingColor(critique.correctness_rating)}>C: {critique.correctness_rating}</span>
                            <span className={getRatingColor(critique.impact_rating)}>I: {critique.impact_rating}</span>
                          </div>
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          ) : (
            <div style={{ 
              display: 'flex', 
              flexDirection: 'column', 
              alignItems: 'center', 
              justifyContent: 'center',
              height: '150px',
              textAlign: 'center',
            }}>
              <div style={{
                padding: '12px',
                backgroundColor: 'rgba(31, 41, 55, 0.5)',
                borderRadius: '50%',
                marginBottom: '12px',
              }}>
                <IconStar className="w-8 h-8 text-gray-600" />
              </div>
              <h3 style={{ fontSize: '14px', fontWeight: '500', color: '#d1d5db', marginBottom: '6px' }}>No Critique Yet</h3>
              <p style={{ fontSize: '11px', color: '#6b7280', maxWidth: '280px' }}>
                Click "Generate Critique" to have your validator model provide an honest assessment of this paper.
              </p>
            </div>
          )}
        </div>

        {/* Footer - Compact */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '10px 16px',
          borderTop: '1px solid rgba(75, 85, 99, 0.5)',
          backgroundColor: 'rgba(31, 41, 55, 0.3)',
        }}>
          <p style={{ fontSize: '10px', color: '#6b7280' }}>
            {critiques.length > 0 && 'Up to 10 critiques saved'}
          </p>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <button
              onClick={onClose}
              style={{
                padding: '6px 12px',
                fontSize: '12px',
                color: '#d1d5db',
                backgroundColor: 'transparent',
                border: 'none',
                cursor: 'pointer',
                borderRadius: '6px',
                transition: 'all 0.2s',
              }}
              onMouseEnter={(e) => {
                e.target.style.backgroundColor = 'rgba(75, 85, 99, 0.3)';
                e.target.style.color = '#f3f4f6';
              }}
              onMouseLeave={(e) => {
                e.target.style.backgroundColor = 'transparent';
                e.target.style.color = '#d1d5db';
              }}
            >
              Close
            </button>
            <button
              onClick={handleGenerateCritique}
              disabled={generating}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                padding: '6px 12px',
                backgroundColor: generating ? '#6b21a8' : '#9333ea',
                color: 'white',
                border: 'none',
                borderRadius: '6px',
                fontSize: '12px',
                fontWeight: '500',
                cursor: generating ? 'not-allowed' : 'pointer',
                transition: 'background-color 0.2s',
              }}
              onMouseEnter={(e) => {
                if (!generating) e.target.style.backgroundColor = '#a855f7';
              }}
              onMouseLeave={(e) => {
                if (!generating) e.target.style.backgroundColor = '#9333ea';
              }}
            >
              {generating ? (
                <>
                  <div style={{
                    width: '12px',
                    height: '12px',
                    border: '2px solid white',
                    borderTopColor: 'transparent',
                    borderRadius: '50%',
                    animation: 'spin 1s linear infinite',
                  }}></div>
                  Generating...
                </>
              ) : (
                <>
                  <IconRefresh className="w-3 h-3" />
                  {selectedCritique ? 'Regenerate' : 'Generate Critique'}
                </>
              )}
            </button>
          </div>
        </div>
      </div>
      
      {/* Keyframes for spinner animation */}
      <style>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );

  // Render via portal to document.body to bypass any parent stacking contexts
  return createPortal(modalContent, document.body);
}

/**
 * Compact rating display component for the modal
 */
function CompactRating({ label, rating, feedback }) {
  const percentage = (rating / 10) * 100;
  
  return (
    <div style={{
      backgroundColor: 'rgba(31, 41, 55, 0.5)',
      borderRadius: '8px',
      padding: '8px 10px',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '4px' }}>
        <span style={{ fontSize: '10px', fontWeight: '500', color: '#d1d5db' }}>{label}</span>
        <span className={getRatingColor(rating)} style={{ fontSize: '14px', fontWeight: '700' }}>
          {rating > 0 ? rating : '—'}
        </span>
      </div>
      
      {/* Progress bar */}
      <div style={{
        height: '4px',
        backgroundColor: 'rgba(55, 65, 81, 1)',
        borderRadius: '9999px',
        overflow: 'hidden',
        marginBottom: feedback ? '6px' : '0',
      }}>
        <div 
          className={getRatingBgColor(rating)}
          style={{ 
            height: '100%', 
            width: `${percentage}%`,
            borderRadius: '9999px',
            transition: 'width 0.5s',
          }}
        />
      </div>
      
      {/* Feedback text - full display */}
      {feedback && (
        <p style={{ 
          fontSize: '11px', 
          color: '#9ca3af', 
          lineHeight: '1.5',
          maxHeight: '250px',
          overflowY: 'auto',
        }}>{feedback}</p>
      )}
    </div>
  );
}

