/**
 * Model cache utility - loads pre-cached model mappings
 * Allows profile system to convert display names to API IDs
 */

let modelCache = null;

/**
 * Load model cache from backend data folder
 */
export async function loadModelCache() {
  if (modelCache) return modelCache;
  
  try {
    const response = await fetch('/api/model-cache');
    if (!response.ok) {
      console.warn('Failed to load model cache:', response.statusText);
      return {};
    }
    
    modelCache = await response.json();
    console.log(`✓ Loaded ${Object.keys(modelCache).length} model mappings`);
    return modelCache;
  } catch (err) {
    console.error('Failed to load model cache:', err);
    return {};
  }
}

/**
 * Convert display name to OpenRouter API ID
 * @param {string} displayNameOrId - The model display name or API ID
 * @returns {string} The OpenRouter API ID (e.g., 'openai/gpt-oss-120b')
 */
export function getModelApiId(displayNameOrId) {
  if (!displayNameOrId) return '';
  
  // If no cache loaded, return as-is
  if (!modelCache) {
    return displayNameOrId;
  }
  
  // Look up in cache
  const cached = modelCache[displayNameOrId];
  if (cached) {
    return cached;
  }
  
  // If not found, assume it's already an API ID
  return displayNameOrId;
}

/**
 * Clear the cache (for testing)
 */
export function clearModelCache() {
  modelCache = null;
}

