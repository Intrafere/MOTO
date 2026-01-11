import React, { useState, useEffect } from 'react';
import AggregatorInterface from './components/aggregator/AggregatorInterface';
import AggregatorSettings from './components/aggregator/AggregatorSettings';
import AggregatorLogs from './components/aggregator/AggregatorLogs';
import LiveResults from './components/aggregator/LiveResults';
import CompilerInterface from './components/compiler/CompilerInterface';
import CompilerSettings from './components/compiler/CompilerSettings';
import CompilerLogs from './components/compiler/CompilerLogs';
import LivePaper from './components/compiler/LivePaper';
import {
  AutonomousResearchInterface,
  BrainstormList,
  PaperLibrary,
  AutonomousResearchSettings,
  AutonomousResearchLogs,
  FinalAnswerView,
  FinalAnswerLibrary
} from './components/autonomous';
import WorkflowPanel from './components/WorkflowPanel';
import BoostControlModal from './components/BoostControlModal';
import BoostLogs from './components/BoostLogs';
import OpenRouterApiKeyModal from './components/OpenRouterApiKeyModal';
import { websocket } from './services/websocket';
import { api, autonomousAPI, openRouterAPI } from './services/api';

function App() {
  const [activeTab, setActiveTab] = useState('auto-interface');
  
  // Single Paper Writer expandable section state
  const [showSinglePaperWriter, setShowSinglePaperWriter] = useState(() => {
    const saved = localStorage.getItem('singlePaperWriterExpanded');
    return saved ? JSON.parse(saved) : false;
  });

  const [singlePaperWriterActiveTab, setSinglePaperWriterActiveTab] = useState('aggregator-interface');
  
  // Models list (fetched from API)
  const [models, setModels] = useState([]);
  
  // Boost modal state
  const [showBoostModal, setShowBoostModal] = useState(false);
  
  // OpenRouter API Key modal state
  const [showOpenRouterKeyModal, setShowOpenRouterKeyModal] = useState(false);
  const [openRouterKeyReason, setOpenRouterKeyReason] = useState('setup');
  
  // LM Studio availability state (for determining default provider)
  const [lmStudioAvailable, setLmStudioAvailable] = useState(true);
  const [hasOpenRouterKey, setHasOpenRouterKey] = useState(false);
  
  // Track if any workflow is running (for WorkflowPanel visibility)
  const [anyWorkflowRunning, setAnyWorkflowRunning] = useState(false);
  
  // Track WorkflowPanel collapse state for sliding boost buttons
  const [workflowPanelCollapsed, setWorkflowPanelCollapsed] = useState(() => {
    const savedState = localStorage.getItem('workflow_panel_collapsed');
    return savedState === 'true';
  });
  
  // Initialize config from localStorage or use defaults
  // CRITICAL: Read from 'aggregator_settings' (used by AggregatorSettings component)
  const [config, setConfig] = useState(() => {
    // Try to load from the settings component key first
    const settingsConfig = localStorage.getItem('aggregator_settings');
    if (settingsConfig) {
      try {
        const settings = JSON.parse(settingsConfig);
        return {
          userPrompt: '',
          submitterConfigs: settings.submitterConfigs || [
            { submitterId: 1, provider: 'lm_studio', modelId: '', openrouterProvider: null, lmStudioFallbackId: null, contextWindow: 131072, maxOutputTokens: 25000 },
            { submitterId: 2, provider: 'lm_studio', modelId: '', openrouterProvider: null, lmStudioFallbackId: null, contextWindow: 131072, maxOutputTokens: 25000 },
            { submitterId: 3, provider: 'lm_studio', modelId: '', openrouterProvider: null, lmStudioFallbackId: null, contextWindow: 131072, maxOutputTokens: 25000 }
          ],
          validatorModel: settings.validatorModel || '',
          validatorProvider: settings.validatorProvider || 'lm_studio',
          validatorOpenrouterProvider: settings.validatorOpenrouterProvider || null,
          validatorLmStudioFallback: settings.validatorLmStudioFallback || null,
          validatorContextSize: settings.validatorContextSize || 131072,
          validatorMaxOutput: settings.validatorMaxOutput || 25000,
          uploadedFiles: [],
        };
      } catch (e) {
        console.error('Failed to parse aggregator_settings:', e);
      }
    }
    
    // Fallback to old key for backward compatibility
    const savedConfig = localStorage.getItem('aggregatorConfig');
    if (savedConfig) {
      try {
        const parsed = JSON.parse(savedConfig);
        return {
          userPrompt: '',
          submitterConfigs: parsed.submitterConfigs || [
            { submitterId: 1, provider: 'lm_studio', modelId: '', openrouterProvider: null, lmStudioFallbackId: null, contextWindow: 131072, maxOutputTokens: 25000 },
            { submitterId: 2, provider: 'lm_studio', modelId: '', openrouterProvider: null, lmStudioFallbackId: null, contextWindow: 131072, maxOutputTokens: 25000 },
            { submitterId: 3, provider: 'lm_studio', modelId: '', openrouterProvider: null, lmStudioFallbackId: null, contextWindow: 131072, maxOutputTokens: 25000 }
          ],
          validatorModel: parsed.validatorModel || '',
          validatorProvider: parsed.validatorProvider || 'lm_studio',
          validatorOpenrouterProvider: parsed.validatorOpenrouterProvider || null,
          validatorLmStudioFallback: parsed.validatorLmStudioFallback || null,
          validatorContextSize: parsed.validatorContextSize || 131072,
          validatorMaxOutput: parsed.validatorMaxOutput || 25000,
          uploadedFiles: [],
        };
      } catch (e) {
        console.error('Failed to parse saved config:', e);
      }
    }
    return {
      userPrompt: '',
      submitterConfigs: [
        { submitterId: 1, provider: 'lm_studio', modelId: '', openrouterProvider: null, lmStudioFallbackId: null, contextWindow: 131072, maxOutputTokens: 25000 },
        { submitterId: 2, provider: 'lm_studio', modelId: '', openrouterProvider: null, lmStudioFallbackId: null, contextWindow: 131072, maxOutputTokens: 25000 },
        { submitterId: 3, provider: 'lm_studio', modelId: '', openrouterProvider: null, lmStudioFallbackId: null, contextWindow: 131072, maxOutputTokens: 25000 }
      ],
      validatorModel: '',
      validatorProvider: 'lm_studio',
      validatorOpenrouterProvider: null,
      validatorLmStudioFallback: null,
      validatorContextSize: 131072,
      validatorMaxOutput: 25000,
      uploadedFiles: [],
    };
  });

  // Save config to localStorage whenever it changes (excluding transient data)
  // CRITICAL: Save to BOTH keys to maintain backward compatibility
  useEffect(() => {
    const configToSave = {
      submitterConfigs: config.submitterConfigs,
      validatorModel: config.validatorModel,
      validatorProvider: config.validatorProvider,
      validatorOpenrouterProvider: config.validatorOpenrouterProvider,
      validatorLmStudioFallback: config.validatorLmStudioFallback,
      validatorContextSize: config.validatorContextSize,
      validatorMaxOutput: config.validatorMaxOutput,
    };
    // Save to both old and new keys
    localStorage.setItem('aggregatorConfig', JSON.stringify(configToSave));
    localStorage.setItem('aggregator_settings', JSON.stringify(configToSave));
  }, [config.submitterConfigs, config.validatorModel, config.validatorProvider, config.validatorOpenrouterProvider, config.validatorLmStudioFallback, config.validatorContextSize, config.validatorMaxOutput]);

  // Autonomous mode state
  const [autonomousRunning, setAutonomousRunning] = useState(false);
  const [autonomousStatus, setAutonomousStatus] = useState(null);
  const [autonomousActivity, setAutonomousActivity] = useState([]);
  const [brainstorms, setBrainstorms] = useState([]);
  const [papers, setPapers] = useState([]);
  const [autonomousStats, setAutonomousStats] = useState(null);
  
  // Disclaimer modal state (shows on every app load)
  const [showDisclaimer, setShowDisclaimer] = useState(true);

  // Autonomous config with localStorage persistence
  // CRITICAL: Read from 'autonomous_research_settings' (used by AutonomousResearchSettings component)
  const [autonomousConfig, setAutonomousConfig] = useState(() => {
    // Try to load from the settings component key first
    const settingsConfig = localStorage.getItem('autonomous_research_settings');
    if (settingsConfig) {
      try {
        const settings = JSON.parse(settingsConfig);
        const localConfig = settings.localConfig || {};
        return {
          submitter_configs: settings.submitterConfigs || [
            { submitterId: 1, provider: 'lm_studio', modelId: '', openrouterProvider: null, lmStudioFallbackId: null, contextWindow: 131072, maxOutputTokens: 25000 },
            { submitterId: 2, provider: 'lm_studio', modelId: '', openrouterProvider: null, lmStudioFallbackId: null, contextWindow: 131072, maxOutputTokens: 25000 },
            { submitterId: 3, provider: 'lm_studio', modelId: '', openrouterProvider: null, lmStudioFallbackId: null, contextWindow: 131072, maxOutputTokens: 25000 }
          ],
          validator_provider: localConfig.validator_provider,
          validator_model: localConfig.validator_model,
          validator_openrouter_provider: localConfig.validator_openrouter_provider,
          validator_lm_studio_fallback: localConfig.validator_lm_studio_fallback,
          validator_context_window: localConfig.validator_context_window,
          validator_max_tokens: localConfig.validator_max_tokens,
          high_context_provider: localConfig.high_context_provider,
          high_context_model: localConfig.high_context_model,
          high_context_openrouter_provider: localConfig.high_context_openrouter_provider,
          high_context_lm_studio_fallback: localConfig.high_context_lm_studio_fallback,
          high_context_context_window: localConfig.high_context_context_window,
          high_context_max_tokens: localConfig.high_context_max_tokens,
          high_param_provider: localConfig.high_param_provider,
          high_param_model: localConfig.high_param_model,
          high_param_openrouter_provider: localConfig.high_param_openrouter_provider,
          high_param_lm_studio_fallback: localConfig.high_param_lm_studio_fallback,
          high_param_context_window: localConfig.high_param_context_window,
          high_param_max_tokens: localConfig.high_param_max_tokens,
          critique_submitter_provider: localConfig.critique_submitter_provider,
          critique_submitter_model: localConfig.critique_submitter_model,
          critique_submitter_openrouter_provider: localConfig.critique_submitter_openrouter_provider,
          critique_submitter_lm_studio_fallback: localConfig.critique_submitter_lm_studio_fallback,
          critique_submitter_context_window: localConfig.critique_submitter_context_window,
          critique_submitter_max_tokens: localConfig.critique_submitter_max_tokens
        };
      } catch (e) {
        console.error('Failed to parse autonomous_research_settings:', e);
      }
    }
    
    // Final fallback - use ACTUAL working defaults (OpenRouter API IDs)
    return {
      submitter_configs: [
        { submitterId: 1, provider: 'openrouter', modelId: 'openai/gpt-oss-120b', openrouterProvider: 'Google', lmStudioFallbackId: null, contextWindow: 131072, maxOutputTokens: 25000 },
        { submitterId: 2, provider: 'openrouter', modelId: 'openai/gpt-oss-20b', openrouterProvider: 'Groq', lmStudioFallbackId: null, contextWindow: 131072, maxOutputTokens: 25000 },
        { submitterId: 3, provider: 'openrouter', modelId: 'openai/gpt-oss-120b', openrouterProvider: 'Google', lmStudioFallbackId: null, contextWindow: 131072, maxOutputTokens: 25000 }
      ],
      validator_provider: 'openrouter',
      validator_model: 'openai/gpt-oss-120b',
      validator_openrouter_provider: 'Google',
      validator_lm_studio_fallback: null,
      validator_context_window: 131072,
      validator_max_tokens: 25000,
      high_context_provider: 'openrouter',
      high_context_model: 'openai/gpt-oss-120b',
      high_context_openrouter_provider: 'Google',
      high_context_lm_studio_fallback: null,
      high_context_context_window: 131072,
      high_context_max_tokens: 25000,
      high_param_provider: 'openrouter',
      high_param_model: 'openai/gpt-oss-120b',
      high_param_openrouter_provider: 'Google',
      high_param_lm_studio_fallback: null,
      high_param_context_window: 131072,
      high_param_max_tokens: 25000,
      critique_submitter_provider: 'openrouter',
      critique_submitter_model: 'openai/gpt-oss-120b',
      critique_submitter_openrouter_provider: 'Google',
      critique_submitter_lm_studio_fallback: null,
      critique_submitter_context_window: 131072,
      critique_submitter_max_tokens: 25000
    };
  });

  // Save autonomous config to localStorage
  // CRITICAL: Save to BOTH keys to maintain backward compatibility
  useEffect(() => {
    localStorage.setItem('autonomousConfig', JSON.stringify(autonomousConfig));
    // Also save to autonomous_research_settings in the format expected by AutonomousResearchSettings
    const settingsToSave = {
      numSubmitters: autonomousConfig.submitter_configs?.length || 3,
      submitterConfigs: autonomousConfig.submitter_configs || [],
      localConfig: {
        validator_provider: autonomousConfig.validator_provider,
        validator_model: autonomousConfig.validator_model,
        validator_openrouter_provider: autonomousConfig.validator_openrouter_provider,
        validator_lm_studio_fallback: autonomousConfig.validator_lm_studio_fallback,
        validator_context_window: autonomousConfig.validator_context_window,
        validator_max_tokens: autonomousConfig.validator_max_tokens,
        high_context_provider: autonomousConfig.high_context_provider,
        high_context_model: autonomousConfig.high_context_model,
        high_context_openrouter_provider: autonomousConfig.high_context_openrouter_provider,
        high_context_lm_studio_fallback: autonomousConfig.high_context_lm_studio_fallback,
        high_context_context_window: autonomousConfig.high_context_context_window,
        high_context_max_tokens: autonomousConfig.high_context_max_tokens,
        high_param_provider: autonomousConfig.high_param_provider,
        high_param_model: autonomousConfig.high_param_model,
        high_param_openrouter_provider: autonomousConfig.high_param_openrouter_provider,
        high_param_lm_studio_fallback: autonomousConfig.high_param_lm_studio_fallback,
        high_param_context_window: autonomousConfig.high_param_context_window,
        high_param_max_tokens: autonomousConfig.high_param_max_tokens,
        critique_submitter_provider: autonomousConfig.critique_submitter_provider,
        critique_submitter_model: autonomousConfig.critique_submitter_model,
        critique_submitter_openrouter_provider: autonomousConfig.critique_submitter_openrouter_provider,
        critique_submitter_lm_studio_fallback: autonomousConfig.critique_submitter_lm_studio_fallback,
        critique_submitter_context_window: autonomousConfig.critique_submitter_context_window,
        critique_submitter_max_tokens: autonomousConfig.critique_submitter_max_tokens
      },
      freeOnly: false // Default value
    };
    localStorage.setItem('autonomous_research_settings', JSON.stringify(settingsToSave));
  }, [autonomousConfig]);

  // Check LM Studio availability and fetch models on mount
  useEffect(() => {
    const checkAvailability = async () => {
      try {
        // Check LM Studio availability
        const lmResult = await openRouterAPI.checkLMStudioAvailability();
        const lmAvailable = lmResult.available && lmResult.has_models;
        setLmStudioAvailable(lmAvailable);
        
        // Check if OpenRouter API key is configured
        const keyStatus = await openRouterAPI.getApiKeyStatus();
        setHasOpenRouterKey(keyStatus.has_key);
        
        // Also check localStorage for saved key and sync with backend
        const storedKey = localStorage.getItem('openrouter_api_key');
        if (storedKey && !keyStatus.has_key) {
          // Restore key to backend from localStorage
          try {
            await openRouterAPI.setApiKey(storedKey);
            setHasOpenRouterKey(true);
          } catch (err) {
            console.error('Failed to restore OpenRouter key:', err);
            localStorage.removeItem('openrouter_api_key');
          }
        }
        
        // If LM Studio not available and no OpenRouter key, prompt for key
        if (!lmAvailable && !keyStatus.has_key && !storedKey) {
          console.log('LM Studio not available, prompting for OpenRouter API key...');
          setOpenRouterKeyReason('lm_studio_unavailable');
          setShowOpenRouterKeyModal(true);
        }
        
        // Fetch LM Studio models if available
        if (lmAvailable) {
          api.getModels().then(data => {
            setModels(data.models || data);
          }).catch(err => {
            console.error('Failed to fetch LM Studio models:', err);
          });
        }
      } catch (err) {
        console.error('Failed to check availability:', err);
        // Fallback to fetching models directly
        api.getModels().then(data => {
          setModels(data.models || data);
        }).catch(modelErr => {
          console.error('Failed to fetch models:', modelErr);
        });
      }
    };
    
    checkAvailability();
  }, []);

  // Check autonomous research status on mount (handles page refresh while running)
  // CRITICAL: Always load all data (brainstorms, papers, stats) on startup,
  // even when not running. This ensures users see existing data immediately
  // without having to click Start first.
  useEffect(() => {
    const checkInitialStatus = async () => {
      try {
        const [status, brainstormsData, papersData, stats] = await Promise.all([
          autonomousAPI.getStatus(),
          autonomousAPI.getBrainstorms(),
          autonomousAPI.getPapers(),
          autonomousAPI.getStats()
        ]);
        
        // ALWAYS load brainstorms, papers, and stats regardless of running state
        // This ensures data is visible on app startup without needing to click Start
        setBrainstorms(brainstormsData.brainstorms || []);
        setPapers(papersData.papers || []);
        setAutonomousStats(stats);
        setAutonomousStatus(status);
        
        // If backend reports running, also sync the running state
        if (status.is_running) {
          console.log('Autonomous research detected as running, syncing state...');
          setAutonomousRunning(true);
        }
      } catch (error) {
        console.error('Failed to check initial autonomous status:', error);
      }
    };
    
    checkInitialStatus();
  }, []);

  // WebSocket connection
  useEffect(() => {
    // Connect to WebSocket
    websocket.connect();

    return () => {
      websocket.disconnect();
    };
  }, []);

  // Autonomous WebSocket event listeners
  useEffect(() => {
    const unsubscribers = [];
    
    // Helper to add activity with limit (prevents unbounded array growth causing UI freeze)
    const MAX_ACTIVITY_EVENTS = 500;
    const addActivity = (event) => {
      setAutonomousActivity(prev => [...prev, event].slice(-MAX_ACTIVITY_EVENTS));
    };
    
    // Topic selection events
    unsubscribers.push(websocket.on('topic_selected', (data) => {
      addActivity({
        event: 'topic_selected',
        timestamp: new Date().toISOString(),
        message: `Topic selected: ${data.topic_prompt}`,
        data
      });
    }));
    
    unsubscribers.push(websocket.on('topic_selection_rejected', (data) => {
      addActivity({
        event: 'topic_selection_rejected',
        timestamp: new Date().toISOString(),
        message: `Topic selection rejected`,
        data
      });
    }));
    
    // Aggregator's direct submission events (per-submission with individual submitter_id)
    unsubscribers.push(websocket.on('submission_accepted', (data) => {
      const modelName = data.submitter_model ? (data.submitter_model.split('/')[1] || data.submitter_model.substring(0, 15)) : 'N/A';
      addActivity({
        event: 'submission_accepted',
        timestamp: new Date().toISOString(),
        message: `Submitter ${data.submitter_id} [${modelName}]: ✓ ACCEPTED (total: ${data.total_acceptances})`,
        data
      });
    }));
    
    unsubscribers.push(websocket.on('submission_rejected', (data) => {
      const modelName = data.submitter_model ? (data.submitter_model.split('/')[1] || data.submitter_model.substring(0, 15)) : 'N/A';
      addActivity({
        event: 'submission_rejected',
        timestamp: new Date().toISOString(),
        message: `Submitter ${data.submitter_id} [${modelName}]: ✗ REJECTED (total: ${data.total_rejections})`,
        data
      });
    }));
    
    // Completion review events
    unsubscribers.push(websocket.on('completion_review_started', (data) => {
      addActivity({
        event: 'completion_review_started',
        timestamp: new Date().toISOString(),
        message: `Completion review started`,
        data
      });
    }));
    
    unsubscribers.push(websocket.on('completion_review_result', (data) => {
      addActivity({
        event: 'completion_review_result',
        timestamp: new Date().toISOString(),
        message: `Decision: ${data.decision}`,
        data
      });
    }));
    
    unsubscribers.push(websocket.on('manual_paper_writing_triggered', (data) => {
      addActivity({
        event: 'manual_paper_writing_triggered',
        timestamp: new Date().toISOString(),
        message: `Manual override: Forcing paper writing for ${data.topic_id} (${data.submission_count} submissions)`,
        data
      });
    }));
    
    // Paper events
    unsubscribers.push(websocket.on('paper_writing_started', (data) => {
      addActivity({
        event: 'paper_writing_started',
        timestamp: new Date().toISOString(),
        message: `Paper writing started: ${data.title}`,
        data
      });
    }));
    
    // Critique phase events (paper writing substages)
    unsubscribers.push(websocket.on('critique_phase_started', (data) => {
      addActivity({
        event: 'critique_phase_started',
        timestamp: new Date().toISOString(),
        message: `Critique phase started (Paper v${data.paper_version || '?'}, target: ${data.target_critiques || 5} critiques)`,
        data
      });
    }));
    
    unsubscribers.push(websocket.on('critique_progress', (data) => {
      // Only log every few updates to avoid spam
      if (data.total_attempts % 2 === 0 || data.total_attempts >= data.target) {
        addActivity({
          event: 'critique_progress',
          timestamp: new Date().toISOString(),
          message: `Critique progress: ${data.acceptances} accepted, ${data.rejections} rejected (${data.total_attempts}/${data.target} attempts)`,
          data
        });
      }
    }));
    
    unsubscribers.push(websocket.on('body_rewrite_started', (data) => {
      addActivity({
        event: 'body_rewrite_started',
        timestamp: new Date().toISOString(),
        message: `REWRITE PHASE: Total rewrite started for Paper v${data.version || '?'}${data.title_changed ? ' (Title updated)' : ''}`,
        data
      });
    }));
    
    unsubscribers.push(websocket.on('partial_revision_complete', (data) => {
      addActivity({
        event: 'partial_revision_complete',
        timestamp: new Date().toISOString(),
        message: `PARTIAL REVISION: Applied ${data.edits_applied || 0} targeted edits (Paper v${data.version || '?'})${data.title_changed ? ' (Title updated)' : ''}`,
        data
      });
    }));
    
    unsubscribers.push(websocket.on('critique_phase_ended', (data) => {
      addActivity({
        event: 'critique_phase_ended',
        timestamp: new Date().toISOString(),
        message: `Critique phase complete (${data.decision || 'unknown'})`,
        data
      });
    }));
    
    unsubscribers.push(websocket.on('critique_phase_skipped', (data) => {
      addActivity({
        event: 'critique_phase_skipped',
        timestamp: new Date().toISOString(),
        message: `Critique phase skipped: ${data.reason || 'user override'}`,
        data
      });
    }));
    
    // Phase transitions during paper writing
    unsubscribers.push(websocket.on('phase_transition', (data) => {
      const fromPhase = data.from_phase || '?';
      const toPhase = data.to_phase || '?';
      const trigger = data.trigger || 'complete';
      addActivity({
        event: 'phase_transition',
        timestamp: new Date().toISOString(),
        message: `Phase transition: ${fromPhase} → ${toPhase} (${trigger})`,
        data
      });
    }));
    
    unsubscribers.push(websocket.on('paper_completed', (data) => {
      addActivity({
        event: 'paper_completed',
        timestamp: new Date().toISOString(),
        message: `Paper completed: ${data.title}`,
        data
      });
      // Refresh papers list
      autonomousAPI.getPapers().then(res => setPapers(res.papers || [])).catch(console.error);
    }));
    
    unsubscribers.push(websocket.on('paper_redundancy_review', (data) => {
      addActivity({
        event: 'paper_redundancy_review',
        timestamp: new Date().toISOString(),
        message: `Redundancy review: ${data.should_remove ? 'Removing paper' : 'No removal'}`,
        data
      });
    }));
    
    unsubscribers.push(websocket.on('auto_research_started', () => {
      setAutonomousActivity([]);
      setAutonomousRunning(true);
    }));
    
    unsubscribers.push(websocket.on('auto_research_resumed', (data) => {
      // Handle resume after crash/restart - sync running state
      console.log('Autonomous research resumed:', data);
      setAutonomousRunning(true);
      addActivity({
        event: 'auto_research_resumed',
        timestamp: new Date().toISOString(),
        message: `Research resumed (${data?.tier || 'unknown tier'})`,
        data
      });
      // Fetch latest status
      autonomousAPI.getStatus().then(status => {
        setAutonomousStatus(status);
      }).catch(console.error);
    }));
    
    unsubscribers.push(websocket.on('auto_research_stopped', () => {
      setAutonomousRunning(false);
    }));
    
    // Tier 3 events
    unsubscribers.push(websocket.on('tier3_started', (data) => {
      addActivity({
        event: 'tier3_started',
        timestamp: new Date().toISOString(),
        message: `Tier 3 Final Answer generation started`,
        data
      });
      // Refresh status to update tier3 info
      autonomousAPI.getStatus().then(setAutonomousStatus).catch(console.error);
    }));
    
    // tier3_result - Backend sends this for certainty assessment result
    unsubscribers.push(websocket.on('tier3_result', (data) => {
      let message;
      if (data.result === 'continue_research') {
        // AI determined that existing papers don't provide a definitive answer yet
        message = 'AI assessment: No definitive answer can be derived from current research. Resuming autonomous research to generate more papers before attempting final answer again.';
      } else {
        message = `Certainty assessment: ${data.certainty_level || 'complete'} - Proceeding to generate final answer`;
      }
      addActivity({
        event: 'tier3_result',
        timestamp: new Date().toISOString(),
        message,
        data
      });
    }));
    
    unsubscribers.push(websocket.on('tier3_format_selected', (data) => {
      addActivity({
        event: 'tier3_format_selected',
        timestamp: new Date().toISOString(),
        message: `Answer format: ${data.format === 'short_form' ? 'Short Form (Single Paper)' : 'Long Form (Volume)'}`,
        data
      });
    }));
    
    // tier3_volume_organized - Backend sends this event name (not tier3_volume_organization_complete)
    unsubscribers.push(websocket.on('tier3_volume_organized', (data) => {
      addActivity({
        event: 'tier3_volume_organized',
        timestamp: new Date().toISOString(),
        message: `Volume organized: "${data.title}" (${data.chapters?.length || 0} chapters)`,
        data
      });
    }));
    
    unsubscribers.push(websocket.on('tier3_chapter_started', (data) => {
      addActivity({
        event: 'tier3_chapter_started',
        timestamp: new Date().toISOString(),
        message: `Writing chapter ${data.chapter_order}: ${data.title}`,
        data
      });
    }));
    
    unsubscribers.push(websocket.on('tier3_chapter_complete', (data) => {
      addActivity({
        event: 'tier3_chapter_complete',
        timestamp: new Date().toISOString(),
        message: `Chapter ${data.chapter_order} complete: ${data.title}`,
        data
      });
    }));
    
    unsubscribers.push(websocket.on('tier3_rejection', (data) => {
      addActivity({
        event: 'tier3_rejection',
        timestamp: new Date().toISOString(),
        message: `Tier 3 submission rejected: ${data.phase || 'unknown phase'}`,
        data
      });
    }));
    
    unsubscribers.push(websocket.on('tier3_complete', (data) => {
      addActivity({
        event: 'tier3_complete',
        timestamp: new Date().toISOString(),
        message: `🏆 FINAL ANSWER COMPLETE! ${data.format === 'short_form' ? 'Paper' : 'Volume'}: "${data.title}"`,
        data
      });
      // Refresh status to update tier3 info
      autonomousAPI.getStatus().then(setAutonomousStatus).catch(console.error);
    }));
    
    // Reference selection events
    unsubscribers.push(websocket.on('reference_selection_started', (data) => {
      addActivity({
        event: 'reference_selection_started',
        timestamp: new Date().toISOString(),
        message: `Reference selection started (${data.mode})`,
        data
      });
    }));
    
    unsubscribers.push(websocket.on('reference_selection_complete', (data) => {
      addActivity({
        event: 'reference_selection_complete',
        timestamp: new Date().toISOString(),
        message: `Reference selection complete: ${data.selected_count} papers selected`,
        data
      });
    }));
    
    // Paper writing resumed (after crash recovery)
    unsubscribers.push(websocket.on('paper_writing_resumed', (data) => {
      addActivity({
        event: 'paper_writing_resumed',
        timestamp: new Date().toISOString(),
        message: `Paper writing resumed: ${data.title}`,
        data
      });
    }));
    
    // Tier 3 additional events
    unsubscribers.push(websocket.on('tier3_forced', (data) => {
      addActivity({
        event: 'tier3_forced',
        timestamp: new Date().toISOString(),
        message: `Tier 3 forced with mode: ${data.mode} (${data.completed_papers} papers available)`,
        data
      });
    }));
    
    unsubscribers.push(websocket.on('tier3_phase_changed', (data) => {
      addActivity({
        event: 'tier3_phase_changed',
        timestamp: new Date().toISOString(),
        message: `Tier 3 phase: ${data.description || data.phase}`,
        data
      });
    }));
    
    unsubscribers.push(websocket.on('tier3_paper_started', (data) => {
      addActivity({
        event: 'tier3_paper_started',
        timestamp: new Date().toISOString(),
        message: `Writing final answer paper: ${data.title}`,
        data
      });
    }));
    
    unsubscribers.push(websocket.on('tier3_short_form_complete', (data) => {
      addActivity({
        event: 'tier3_short_form_complete',
        timestamp: new Date().toISOString(),
        message: `Short form paper complete: ${data.title}`,
        data
      });
    }));
    
    unsubscribers.push(websocket.on('tier3_long_form_complete', (data) => {
      addActivity({
        event: 'tier3_long_form_complete',
        timestamp: new Date().toISOString(),
        message: `Long form volume complete: ${data.title} (${data.total_chapters} chapters)`,
        data
      });
    }));
    
    unsubscribers.push(websocket.on('final_answer_complete', (data) => {
      addActivity({
        event: 'final_answer_complete',
        timestamp: new Date().toISOString(),
        message: `Final answer complete! Format: ${data.format}`,
        data
      });
      // Refresh status
      autonomousAPI.getStatus().then(setAutonomousStatus).catch(console.error);
    }));
    
    return () => {
      unsubscribers.forEach(unsub => unsub());
    };
  }, []);

  // Poll for autonomous data while running
  useEffect(() => {
    if (!autonomousRunning) return;
    
    const interval = setInterval(async () => {
      try {
        const [status, brainstormsData, papersData, stats] = await Promise.all([
          autonomousAPI.getStatus(),
          autonomousAPI.getBrainstorms(),
          autonomousAPI.getPapers(),
          autonomousAPI.getStats()
        ]);
        
        setAutonomousStatus(status);
        setBrainstorms(brainstormsData.brainstorms || []);
        setPapers(papersData.papers || []);
        setAutonomousStats(stats);
      } catch (error) {
        console.error('Failed to poll autonomous data:', error);
      }
    }, 3000);
    
    return () => clearInterval(interval);
  }, [autonomousRunning]);

  // Autonomous handlers
  const handleAutonomousStart = async (researchPrompt) => {
    try {
      // Convert frontend camelCase to backend snake_case for submitter_configs (includes OpenRouter fields)
      const submitterConfigs = autonomousConfig.submitter_configs?.map(cfg => ({
        submitter_id: cfg.submitterId,
        provider: cfg.provider || 'lm_studio',
        model_id: cfg.modelId,
        openrouter_provider: cfg.openrouterProvider || null,
        lm_studio_fallback_id: cfg.lmStudioFallbackId || null,
        context_window: cfg.contextWindow,
        max_output_tokens: cfg.maxOutputTokens
      })) || [];

      await autonomousAPI.start({
        user_research_prompt: researchPrompt,
        submitter_configs: submitterConfigs,
        // Validator config with OpenRouter support
        validator_provider: autonomousConfig.validator_provider,
        validator_model: autonomousConfig.validator_model,
        validator_openrouter_provider: autonomousConfig.validator_openrouter_provider,
        validator_lm_studio_fallback: autonomousConfig.validator_lm_studio_fallback,
        validator_context_window: autonomousConfig.validator_context_window,
        validator_max_tokens: autonomousConfig.validator_max_tokens,
        // High-context submitter config with OpenRouter support
        high_context_provider: autonomousConfig.high_context_provider,
        high_context_model: autonomousConfig.high_context_model,
        high_context_openrouter_provider: autonomousConfig.high_context_openrouter_provider,
        high_context_lm_studio_fallback: autonomousConfig.high_context_lm_studio_fallback,
        high_context_context_window: autonomousConfig.high_context_context_window,
        high_context_max_tokens: autonomousConfig.high_context_max_tokens,
        // High-param submitter config with OpenRouter support
        high_param_provider: autonomousConfig.high_param_provider,
        high_param_model: autonomousConfig.high_param_model,
        high_param_openrouter_provider: autonomousConfig.high_param_openrouter_provider,
        high_param_lm_studio_fallback: autonomousConfig.high_param_lm_studio_fallback,
        high_param_context_window: autonomousConfig.high_param_context_window,
        high_param_max_tokens: autonomousConfig.high_param_max_tokens,
        // Critique submitter config with OpenRouter support
        critique_submitter_provider: autonomousConfig.critique_submitter_provider,
        critique_submitter_model: autonomousConfig.critique_submitter_model,
        critique_submitter_openrouter_provider: autonomousConfig.critique_submitter_openrouter_provider,
        critique_submitter_lm_studio_fallback: autonomousConfig.critique_submitter_lm_studio_fallback,
        critique_submitter_context_window: autonomousConfig.critique_submitter_context_window,
        critique_submitter_max_tokens: autonomousConfig.critique_submitter_max_tokens
      });
      setAutonomousRunning(true);
      setAutonomousActivity([]);
    } catch (error) {
      alert(`Failed to start autonomous research: ${error.message}`);
    }
  };

  const handleAutonomousStop = async () => {
    try {
      await autonomousAPI.stop();
      setAutonomousRunning(false);
    } catch (error) {
      alert(`Failed to stop autonomous research: ${error.message}`);
    }
  };

  const handleAutonomousClear = async () => {
    if (!window.confirm('Clear all autonomous research data? This cannot be undone.')) {
      return;
    }
    try {
      const result = await autonomousAPI.clear();
      
      // Success - clear frontend state
      setBrainstorms([]);
      setPapers([]);
      setAutonomousActivity([]);
      setAutonomousStats(null);
      
      // Show success message
      if (result.warnings) {
        alert(`Data cleared successfully.\n\nNote: ${result.warnings}`);
      } else {
        alert('All autonomous research data cleared successfully.');
      }
    } catch (error) {
      // Show detailed error message
      const errorMsg = error.details || error.message || 'Unknown error';
      alert(`Failed to clear data:\n\n${errorMsg}\n\nThis may be due to Windows file locking. Try closing file explorer and any programs that may have files open, then try again.`);
    }
  };

  const refreshBrainstorms = async () => {
    try {
      const data = await autonomousAPI.getBrainstorms();
      setBrainstorms(data.brainstorms || []);
    } catch (error) {
      console.error('Failed to refresh brainstorms:', error);
    }
  };

  const refreshPapers = async () => {
    try {
      const data = await autonomousAPI.getPapers();
      setPapers(data.papers || []);
    } catch (error) {
      console.error('Failed to refresh papers:', error);
    }
  };

  // Determine Final Answer tab label based on Tier 3 status
  const getFinalAnswerLabel = () => {
    if (autonomousStatus?.is_tier3_active) {
      return 'Stage 3:FINAL ANSWER IN PROGRESS';
    }
    if (autonomousStatus?.tier3_status === 'complete') {
      return 'Stage 3: FINAL ANSWER COMPLETE ✓';
    }
    return 'Stage 3: Final Answer';
  };

  const mainTabs = [
    { id: 'auto-interface', label: 'Start Here: Autonomous Deep Research Controller', group: 'autonomous-main' },
    { id: 'auto-brainstorms', label: 'Stage 1: Brainstorms', group: 'autonomous-main' },
    { id: 'auto-papers', label: 'Stage 2: Working Final Answer(s)', subtext: '(Less Hallucinatory - Abbreviated Final Answers)', subtextClass: 'green', group: 'autonomous-main' },
    { id: 'auto-final-answer', label: getFinalAnswerLabel(), subtext: '(Very Experimental and Hallucinatory)', group: 'autonomous-main' },
    { id: 'auto-final-answer-library', label: 'Final Answer History', subtext: '(Very Experimental and Hallucinatory)', group: 'autonomous-main' },
  ];

  const autonomousSettingsTabs = [
    { id: 'auto-settings', label: 'Autonomous Model Selection & Settings', group: 'autonomous-settings' },
    { id: 'auto-logs', label: 'API Call Logs', group: 'autonomous-settings' },
  ];

  const singlePaperWriterTabs = {
    aggregator: [
      { id: 'aggregator-interface', label: 'Interface' },
      { id: 'aggregator-settings', label: 'Settings' },
      { id: 'aggregator-logs', label: 'Logs' },
      { id: 'aggregator-results', label: 'Live Results' },
    ],
    compiler: [
      { id: 'compiler-interface', label: 'Interface' },
      { id: 'compiler-settings', label: 'Settings' },
      { id: 'compiler-logs', label: 'Logs' },
      { id: 'compiler-live-paper', label: 'Live Paper' },
    ]
  };

  // Sync with WorkflowPanel collapse state (stored in localStorage)
  useEffect(() => {
    const handleStorageChange = () => {
      const savedState = localStorage.getItem('workflow_panel_collapsed');
      setWorkflowPanelCollapsed(savedState === 'true');
    };
    
    const interval = setInterval(handleStorageChange, 500);
    return () => clearInterval(interval);
  }, []);

  // Check if any workflow is running
  useEffect(() => {
    const checkWorkflowStatus = async () => {
      try {
        const [aggStatus, compStatus, autoStatus] = await Promise.all([
          api.get('/api/aggregator/status').catch(() => ({ is_running: false })),
          api.get('/api/compiler/status').catch(() => ({ is_running: false })),
          autonomousAPI.getStatus().catch(() => ({ is_running: false }))
        ]);
        
        const running = aggStatus.is_running || compStatus.is_running || autoStatus.is_running;
        setAnyWorkflowRunning(running);
      } catch (error) {
        console.error('Failed to check workflow status:', error);
      }
    };
    
    checkWorkflowStatus();
    const interval = setInterval(checkWorkflowStatus, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="app">
      {/* Banner Section */}
      <div className="app-banner">
        <div className="banner-content">
          <h1 className="banner-title">
            <span className="banner-moto">M.O.T.O.</span>
            <span className="banner-subtitle">Deep Research Harness</span>
          </h1>
          <p className="banner-variant">Creative Math Researcher Variant (High Risk, High Reward)</p>
          <p className="banner-company">By Intrafere LLC</p>
        </div>
      </div>
      
      {/* CRITICAL: Boost buttons are ETERNAL - they NEVER disappear */}
      {/* These buttons are fixed-position, high z-index, and unconditionally rendered */}
      {/* They are visible at program launch and stay visible forever */}
      {/* Slide with WorkflowPanel collapse/expand animation */}
      <div className={`app-header ${workflowPanelCollapsed ? 'panel-collapsed' : ''}`}>
        <button 
          className="boost-btn"
          onClick={() => setShowBoostModal(true)}
          title="Configure API Boost"
        >
          ⚡ API Boost
        </button>
        <button 
          className="boost-logs-btn"
          onClick={() => {
            setActiveTab('boost-logs');
            setShowSinglePaperWriter(false);
          }}
          title="View Boost Logs"
        >
          Boost Logs
        </button>
        <button 
          className="openrouter-key-btn"
          onClick={() => {
            setOpenRouterKeyReason('setup');
            setShowOpenRouterKeyModal(true);
          }}
          title="Configure OpenRouter API Key"
          style={{
            marginLeft: '0.5rem',
            padding: '0.4rem 0.8rem',
            backgroundColor: hasOpenRouterKey ? '#2d5a27' : '#4a3a00',
            border: hasOpenRouterKey ? '1px solid #4CAF50' : '1px solid #f1c40f',
            borderRadius: '4px',
            color: '#fff',
            cursor: 'pointer',
            fontSize: '0.85rem',
          }}
        >
          {hasOpenRouterKey ? 'OpenRouter ✓' : 'Set OpenRouter Key'}
        </button>
        {!lmStudioAvailable && (
          <span style={{ 
            marginLeft: '0.5rem', 
            color: '#f1c40f', 
            fontSize: '0.8rem',
            padding: '0.25rem 0.5rem',
            backgroundColor: 'rgba(241, 196, 15, 0.1)',
            borderRadius: '4px',
          }}>
            LM Studio Offline
          </span>
        )}
      </div>
      
      <div className="tabs">
        {mainTabs.map((tab, index) => {
          const prevTab = mainTabs[index - 1];
          const showSeparator = prevTab && prevTab.group !== tab.group;
          
          // Special styling for Final Answer tab
          const isFinalAnswerTab = tab.id === 'auto-final-answer';
          const tier3Classes = isFinalAnswerTab 
            ? (autonomousStatus?.tier3_status === 'complete' 
                ? 'tab-tier3-complete' 
                : (autonomousStatus?.is_tier3_active ? 'tab-tier3-active' : ''))
            : '';
          
          return (
            <React.Fragment key={tab.id}>
              {showSeparator && <div className="tab-separator" />}
              <button
                className={`tab ${activeTab === tab.id ? 'active' : ''} tab-${tab.group} ${tier3Classes} ${tab.subtext ? 'tab-with-subtext' : ''}`}
                onClick={() => {
                  setActiveTab(tab.id);
                  setShowSinglePaperWriter(false);
                }}
              >
                <div className="tab-content-wrapper">
                  <span className="tab-main-label">{tab.label}</span>
                  {tab.subtext && <span className={`tab-subtext ${tab.subtextClass || ''}`}>{tab.subtext}</span>}
                </div>
              </button>
            </React.Fragment>
          );
        })}
        
        {/* Large spacer for settings group */}
        <div className="tab-group-spacer-large"></div>
        
        {autonomousSettingsTabs.map(tab => {
          return (
            <React.Fragment key={tab.id}>
              <button
                className={`tab ${activeTab === tab.id ? 'active' : ''} tab-${tab.group}`}
                onClick={() => {
                  setActiveTab(tab.id);
                  setShowSinglePaperWriter(false);
                }}
              >
                {tab.label}
              </button>
            </React.Fragment>
          );
        })}
      </div>
      
      {/* Expandable Single Paper Writer Section */}
      <div className="expandable-section">
        <button 
          className={`expandable-trigger ${showSinglePaperWriter ? 'expanded' : ''}`}
          onClick={() => {
            const newState = !showSinglePaperWriter;
            setShowSinglePaperWriter(newState);
            localStorage.setItem('singlePaperWriterExpanded', JSON.stringify(newState));
            if (newState && !singlePaperWriterActiveTab) {
              setSinglePaperWriterActiveTab('aggregator-interface');
            }
          }}
        >
          <span className="expand-icon">{showSinglePaperWriter ? '▼' : '▶'}</span>
          <span className="section-title">[Secondary Tool] SINGLE PAPER WRITER</span>
          <span className="section-subtitle">(A manual brainstorm aggregator & paper compiler, an advanced controller mode with a "two user prompts" control mechanic, a separate optional mode from the Autonomous Deep Research mode above)</span>
        </button>
        
        {showSinglePaperWriter && (
          <div className="expandable-content">
            <div className="subsection">
              <div className="subsection-header">AGGREGATOR</div>
              <div className="subsection-tabs">
                {singlePaperWriterTabs.aggregator.map(tab => (
                  <button
                    key={tab.id}
                    className={`subtab ${singlePaperWriterActiveTab === tab.id ? 'active' : ''}`}
                    onClick={() => {
                      setSinglePaperWriterActiveTab(tab.id);
                      setActiveTab(null);
                    }}
                  >
                    {tab.label}
                  </button>
                ))}
              </div>
            </div>
            
            <div className="subsection">
              <div className="subsection-header">COMPILER</div>
              <div className="subsection-tabs">
                {singlePaperWriterTabs.compiler.map(tab => (
                  <button
                    key={tab.id}
                    className={`subtab ${singlePaperWriterActiveTab === tab.id ? 'active' : ''}`}
                    onClick={() => {
                      setSinglePaperWriterActiveTab(tab.id);
                      setActiveTab(null);
                    }}
                  >
                    {tab.label}
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
      
      <div className="tab-content">
        <div className="container">
          {/* Main Tabs Content */}
          {activeTab === 'auto-interface' && (
            <AutonomousResearchInterface
              isRunning={autonomousRunning}
              status={autonomousStatus}
              activity={autonomousActivity}
              onStart={handleAutonomousStart}
              onStop={handleAutonomousStop}
              onClear={handleAutonomousClear}
              config={autonomousConfig}
              api={autonomousAPI}
            />
          )}
          {activeTab === 'auto-brainstorms' && (
            <BrainstormList
              brainstorms={brainstorms}
              onRefresh={refreshBrainstorms}
              api={{ 
                getBrainstorm: autonomousAPI.getBrainstorm,
                deleteBrainstorm: autonomousAPI.deleteBrainstorm
              }}
            />
          )}
          {activeTab === 'auto-papers' && (
            <PaperLibrary
              papers={papers}
              onRefresh={refreshPapers}
              archivedCount={autonomousStats?.paper_counts?.archived || 0}
              api={{ 
                getAutonomousPaper: autonomousAPI.getAutonomousPaper,
                deletePaper: autonomousAPI.deletePaper
              }}
            />
          )}
          {activeTab === 'auto-final-answer' && (
            <FinalAnswerView
              api={autonomousAPI}
              isRunning={autonomousRunning}
              status={autonomousStatus}
            />
          )}
          {activeTab === 'auto-final-answer-library' && (
            <FinalAnswerLibrary />
          )}
          {activeTab === 'auto-logs' && (
            <AutonomousResearchLogs
              stats={autonomousStats}
              events={autonomousActivity}
            />
          )}
          {activeTab === 'boost-logs' && <BoostLogs />}
          
          {/* Single Paper Writer Content - ONLY when section is expanded */}
          {showSinglePaperWriter && singlePaperWriterActiveTab === 'aggregator-interface' && (
            <AggregatorInterface config={config} setConfig={setConfig} />
          )}
          {showSinglePaperWriter && singlePaperWriterActiveTab === 'aggregator-settings' && (
            <AggregatorSettings config={config} setConfig={setConfig} />
          )}
          {showSinglePaperWriter && singlePaperWriterActiveTab === 'aggregator-logs' && <AggregatorLogs />}
          {showSinglePaperWriter && singlePaperWriterActiveTab === 'aggregator-results' && <LiveResults />}
          
          {showSinglePaperWriter && singlePaperWriterActiveTab === 'compiler-interface' && <CompilerInterface activeTab={singlePaperWriterActiveTab} />}
          {showSinglePaperWriter && singlePaperWriterActiveTab === 'compiler-settings' && <CompilerSettings />}
          {showSinglePaperWriter && singlePaperWriterActiveTab === 'compiler-logs' && <CompilerLogs />}
          {showSinglePaperWriter && singlePaperWriterActiveTab === 'compiler-live-paper' && <LivePaper />}
        </div>
      </div>
      
      {/* Autonomous Settings - Rendered OUTSIDE tab-content to allow full-width sidebar layout */}
      {activeTab === 'auto-settings' && (
        <AutonomousResearchSettings
          config={autonomousConfig}
          onConfigChange={setAutonomousConfig}
          models={models}
          isRunning={autonomousRunning}
        />
      )}
      
      {/* WorkflowPanel is ETERNAL - always visible for boost controls */}
      {/* The panel shows workflow tasks when running, but boost controls are ALWAYS accessible */}
      {/* Users can configure boost (set next count, toggle categories) at any time */}
      <WorkflowPanel isRunning={anyWorkflowRunning} />
      
      {/* Beta Disclaimer Modal - Shows on every app load */}
      {showDisclaimer && (
        <>
          <div className="disclaimer-overlay" onClick={(e) => e.stopPropagation()} />
          <div className="disclaimer-modal">
            <div className="disclaimer-content">
              <h2 style={{ marginTop: 0, marginBottom: '1.5rem', color: '#f1c40f' }}>
                  Beta Prototype Warning
              </h2>
              <p style={{ fontSize: '0.95rem', lineHeight: '1.5', marginBottom: '1.5rem' }}>
                Disclaimer: This program is a prototype and in its beta development phase. This NOT meant to produce a single paper, the first paper may lack in quality, MOTO is intended to generate you dozens of papers and improve with each completely new paper, best results show after 10+ papers.
                Watch your systems and API keys for infinite loops, wasted API calls, and any other bugs. This first version of the program is powerful but currently has many bugs. The paper text rendering system is experimental—display issues are <em>not</em> reflective of paper quality. If formatting appears messy, try a 3rd-party LaTeX renderer or copy the raw text into another LLM chat for verification.
              </p>
              <p style={{ fontSize: '1.1rem', lineHeight: '1.6', marginBottom: '1.5rem', color: '#ffcc00' }}>
                <strong>QUICKSTART:</strong> (Optional) Load your Nomic embedding agent on LM STUDIO, or use an OpenRouter API key-only instead of LM STUDIO and go straight to picking your models, and then start the program - expect it to run for at the VERY LEAST hours to days once you hit run. You must leave your PC on and awake during runtime.
              </p>
              <p style={{ fontSize: '0.95rem', lineHeight: '1.5', marginBottom: '1.5rem', color: '#bbb' }}>
                Please report all bugs and issues to project the repo at <a href="https://github.com/Intrafere/MOTO" target="_blank" rel="noopener noreferrer" style={{ color: '#f1c40f', textDecoration: 'none' }}>GitHub</a>.
              </p>
              <p style={{ fontSize: '0.95rem', lineHeight: '1.5', marginBottom: '1.5rem', color: '#bbb' }}>
                Trouble shoot and modify this program easily using the code's specialized rules for AIs and Cursor.com's agentic code editing app - no programming experience required!
              </p>
              <button 
                className="disclaimer-acknowledge-btn"
                onClick={() => setShowDisclaimer(false)}
              >
                Acknowledged
              </button>
            </div>
          </div>
        </>
      )}
      
      {/* Boost Control Modal */}
      <BoostControlModal 
        isOpen={showBoostModal}
        onClose={() => setShowBoostModal(false)}
      />
      
      {/* OpenRouter API Key Modal */}
      <OpenRouterApiKeyModal
        isOpen={showOpenRouterKeyModal}
        onClose={() => setShowOpenRouterKeyModal(false)}
        onKeySet={(key) => {
          setHasOpenRouterKey(true);
          console.log('OpenRouter API key set successfully');
        }}
        reason={openRouterKeyReason}
      />
      
      {/* Footer Section */}
      <footer className="app-footer">
        <div className="footer-content">
          <div className="footer-section footer-license">
            <span>MIT License</span>
            <span className="footer-divider">|</span>
            <span className="footer-copyright">© 2025 Intrafere LLC</span>
          </div>
          
          <div className="footer-section footer-links">
            <a 
              href="https://intrafere.com/moto-autonomous-home-ai/" 
              target="_blank" 
              rel="noopener noreferrer"
              className="footer-link"
            >
              <span className="footer-icon">ℹ️</span>
              About M.O.T.O.
            </a>
            <a 
              href="https://intrafere.com/moto-news/" 
              target="_blank" 
              rel="noopener noreferrer"
              className="footer-link footer-link-news"
            >
              MOTO News and Updates
            </a>
          </div>
          
          <div className="footer-section footer-donate">
            <a 
              href="https://intrafere.com/donate/" 
              target="_blank" 
              rel="noopener noreferrer"
              className="donate-btn"
            >
              Donate - Support open source!
            </a>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;

