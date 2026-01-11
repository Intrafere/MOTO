import React, { useState, useEffect } from 'react';
import { api } from '../../services/api';
import LatexRenderer from '../LatexRenderer';

// Simple inline icon components (no external dependency)
const IconX = ({ className }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <line x1="18" y1="6" x2="6" y2="18"></line>
    <line x1="6" y1="6" x2="18" y2="18"></line>
  </svg>
);
const IconFileText = ({ className }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
    <polyline points="14 2 14 8 20 8"></polyline>
    <line x1="16" y1="13" x2="8" y2="13"></line>
    <line x1="16" y1="17" x2="8" y2="17"></line>
  </svg>
);
const IconDatabase = ({ className }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <ellipse cx="12" cy="5" rx="9" ry="3"></ellipse>
    <path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"></path>
    <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"></path>
  </svg>
);
const IconChevronRight = ({ className }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <polyline points="9 18 15 12 9 6"></polyline>
  </svg>
);

/**
 * Modal overlay for viewing archived research lineage (papers + brainstorms).
 * Read-only view that looks similar to live interface but can't be continued.
 */
export default function ArchiveViewerModal({ answerId, onClose }) {
  const [activeTab, setActiveTab] = useState('papers'); // 'papers' | 'brainstorms'
  const [papers, setPapers] = useState([]);
  const [brainstorms, setBrainstorms] = useState([]);
  const [selectedPaper, setSelectedPaper] = useState(null);
  const [selectedBrainstorm, setSelectedBrainstorm] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadArchive();
  }, [answerId]);

  const loadArchive = async () => {
    try {
      setLoading(true);
      
      // Load papers list
      const papersRes = await api.get(`/auto-research/final-answer/${answerId}/archive/papers`);
      setPapers(papersRes.data.papers);
      
      // Load brainstorms list
      const brainstormsRes = await api.get(`/auto-research/final-answer/${answerId}/archive/brainstorms`);
      setBrainstorms(brainstormsRes.data.brainstorms);
      
    } catch (error) {
      console.error('Failed to load archive:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadPaperDetails = async (paperId) => {
    try {
      const res = await api.get(`/auto-research/final-answer/${answerId}/archive/papers/${paperId}`);
      setSelectedPaper(res.data);
    } catch (error) {
      console.error('Failed to load paper:', error);
    }
  };

  const loadBrainstormDetails = async (topicId) => {
    try {
      const res = await api.get(`/auto-research/final-answer/${answerId}/archive/brainstorms/${topicId}`);
      setSelectedBrainstorm(res.data);
    } catch (error) {
      console.error('Failed to load brainstorm:', error);
    }
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-gray-900 rounded-lg w-11/12 h-5/6 max-w-6xl flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-gray-700">
          <h2 className="text-xl font-semibold text-gray-100 flex items-center gap-2">
            <IconDatabase className="w-5 h-5 text-blue-400" />
            Research Archive (Read-Only)
          </h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-200"
          >
            <IconX className="w-6 h-6" />
          </button>
        </div>

        {/* Tabs */}
        <div className="flex border-b border-gray-700">
          <button
            onClick={() => {
              setActiveTab('papers');
              setSelectedPaper(null);
            }}
            className={`px-6 py-3 font-medium transition-colors ${
              activeTab === 'papers'
                ? 'text-blue-400 border-b-2 border-blue-400'
                : 'text-gray-400 hover:text-gray-200'
            }`}
          >
            <IconFileText className="inline w-4 h-4 mr-2" />
            Papers ({papers.length})
          </button>
          <button
            onClick={() => {
              setActiveTab('brainstorms');
              setSelectedBrainstorm(null);
            }}
            className={`px-6 py-3 font-medium transition-colors ${
              activeTab === 'brainstorms'
                ? 'text-blue-400 border-b-2 border-blue-400'
                : 'text-gray-400 hover:text-gray-200'
            }`}
          >
            <IconDatabase className="inline w-4 h-4 mr-2" />
            Brainstorms ({brainstorms.length})
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-4">
          {loading ? (
            <div className="text-center text-gray-400 py-8">Loading archive...</div>
          ) : (
            <>
              {activeTab === 'papers' && (
                selectedPaper ? (
                  <PaperDetailView paper={selectedPaper} onBack={() => setSelectedPaper(null)} />
                ) : (
                  <PapersListView papers={papers} onSelectPaper={loadPaperDetails} />
                )
              )}
              
              {activeTab === 'brainstorms' && (
                selectedBrainstorm ? (
                  <BrainstormDetailView brainstorm={selectedBrainstorm} onBack={() => setSelectedBrainstorm(null)} />
                ) : (
                  <BrainstormsListView brainstorms={brainstorms} onSelectBrainstorm={loadBrainstormDetails} />
                )
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

// Papers list view
function PapersListView({ papers, onSelectPaper }) {
  if (papers.length === 0) {
    return <div className="text-gray-400 text-center py-8">No papers in archive</div>;
  }

  return (
    <div className="space-y-3">
      {papers.map((paper) => (
        <div
          key={paper.paper_id}
          onClick={() => onSelectPaper(paper.paper_id)}
          className="bg-gray-800 p-4 rounded-lg cursor-pointer hover:bg-gray-750 transition-colors"
        >
          <div className="flex items-start justify-between">
            <div className="flex-1">
              <h3 className="text-lg font-semibold text-gray-100 mb-2">{paper.title}</h3>
              <p className="text-sm text-gray-400 line-clamp-2">{paper.abstract}</p>
              <div className="mt-2 text-xs text-gray-500">
                {paper.word_count} words • Paper ID: {paper.paper_id}
              </div>
            </div>
            <IconChevronRight className="w-5 h-5 text-gray-500 ml-4 flex-shrink-0" />
          </div>
        </div>
      ))}
    </div>
  );
}

// Paper detail view
function PaperDetailView({ paper, onBack }) {
  return (
    <div>
      <button
        onClick={onBack}
        className="text-blue-400 hover:text-blue-300 mb-4 flex items-center gap-1"
      >
        ← Back to Papers
      </button>
      
      <div className="bg-gray-800 p-6 rounded-lg">
        <div className="mb-4 pb-4 border-b border-gray-700">
          <span className="text-xs text-yellow-500 bg-yellow-500/10 px-2 py-1 rounded">
            ARCHIVED - READ ONLY
          </span>
        </div>
        
        <h2 className="text-2xl font-bold text-gray-100 mb-4">{paper.metadata.title}</h2>
        
        <div className="mb-6">
          <h3 className="text-sm font-semibold text-gray-400 mb-2">Abstract</h3>
          <p className="text-gray-300">{paper.abstract}</p>
        </div>
        
        <div className="mb-6">
          <h3 className="text-sm font-semibold text-gray-400 mb-2">Paper Content</h3>
          <div className="bg-gray-900 rounded max-h-96 overflow-y-auto">
            <LatexRenderer
              content={
                paper.outline
                  ? `${paper.outline}\n\n${'='.repeat(80)}\n\n${paper.content}`
                  : paper.content
              }
              className="archive-paper-renderer"
              showToggle={true}
              defaultRaw={false}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

// Brainstorms list view
function BrainstormsListView({ brainstorms, onSelectBrainstorm }) {
  if (brainstorms.length === 0) {
    return <div className="text-gray-400 text-center py-8">No brainstorms in archive</div>;
  }

  return (
    <div className="space-y-3">
      {brainstorms.map((brainstorm) => (
        <div
          key={brainstorm.topic_id}
          onClick={() => onSelectBrainstorm(brainstorm.topic_id)}
          className="bg-gray-800 p-4 rounded-lg cursor-pointer hover:bg-gray-750 transition-colors"
        >
          <div className="flex items-start justify-between">
            <div className="flex-1">
              <h3 className="text-lg font-semibold text-gray-100 mb-2">{brainstorm.topic_prompt}</h3>
              <div className="text-sm text-gray-400">
                {brainstorm.submission_count} submissions • Status: {brainstorm.status}
              </div>
              <div className="mt-2 text-xs text-gray-500">
                Topic ID: {brainstorm.topic_id}
              </div>
            </div>
            <IconChevronRight className="w-5 h-5 text-gray-500 ml-4 flex-shrink-0" />
          </div>
        </div>
      ))}
    </div>
  );
}

// Brainstorm detail view
function BrainstormDetailView({ brainstorm, onBack }) {
  return (
    <div>
      <button
        onClick={onBack}
        className="text-blue-400 hover:text-blue-300 mb-4 flex items-center gap-1"
      >
        ← Back to Brainstorms
      </button>
      
      <div className="bg-gray-800 p-6 rounded-lg">
        <div className="mb-4 pb-4 border-b border-gray-700">
          <span className="text-xs text-yellow-500 bg-yellow-500/10 px-2 py-1 rounded">
            ARCHIVED - READ ONLY
          </span>
        </div>
        
        <h2 className="text-2xl font-bold text-gray-100 mb-4">{brainstorm.metadata.topic_prompt}</h2>
        
        <div className="mb-6 text-sm text-gray-400">
          <div>Status: {brainstorm.metadata.status}</div>
          <div>Submissions: {brainstorm.metadata.submission_count}</div>
          <div>Topic ID: {brainstorm.topic_id}</div>
        </div>
        
        <div>
          <h3 className="text-sm font-semibold text-gray-400 mb-2">Brainstorm Database</h3>
          <pre className="text-gray-300 whitespace-pre-wrap font-mono text-sm bg-gray-900 p-4 rounded max-h-96 overflow-y-auto">
            {brainstorm.content}
          </pre>
        </div>
      </div>
    </div>
  );
}

