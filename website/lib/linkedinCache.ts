import fs from 'fs';
import path from 'path';

/**
 * LinkedIn Profile Cache System
 *
 * Caches LinkedIn profile data to avoid excessive API usage during testing.
 * Cache files are stored in temp/linkedin_cache/ directory.
 */

const CACHE_DIR = path.join(process.cwd(), 'temp', 'linkedin_cache');

interface CachedProfile {
  url: string;
  cachedAt: string;
  profile: any; // LinkedInProfile type
}

/**
 * Extract cache key from LinkedIn URL
 *
 * @param url - LinkedIn profile URL (e.g., "https://linkedin.com/in/bulut-yazgan")
 * @returns Cache key (e.g., "bulut-yazgan") or null if invalid URL
 */
export function getCacheKey(url: string): string | null {
  try {
    // Match LinkedIn username from URL
    const match = url.match(/linkedin\.com\/in\/([^/?]+)/);
    return match ? match[1] : null;
  } catch (error) {
    console.error('Error extracting cache key from URL:', url, error);
    return null;
  }
}

/**
 * Get full file path for cache file
 *
 * @param url - LinkedIn profile URL
 * @returns Absolute path to cache file
 */
export function getCachePath(url: string): string | null {
  const key = getCacheKey(url);
  if (!key) return null;
  return path.join(CACHE_DIR, `${key}.json`);
}

/**
 * Read cached profile data from disk
 *
 * @param url - LinkedIn profile URL
 * @returns Cached profile data or null if not found/invalid
 */
export function readFromCache(url: string): CachedProfile | null {
  const filePath = getCachePath(url);
  if (!filePath) {
    console.error('❌ Invalid LinkedIn URL for cache read:', url);
    return null;
  }

  // Check if cache file exists
  if (!fs.existsSync(filePath)) {
    return null;
  }

  try {
    const data = fs.readFileSync(filePath, 'utf-8');
    const cached: CachedProfile = JSON.parse(data);

    console.log(`📦 Cache hit: ${getCacheKey(url)}.json`);
    return cached;
  } catch (error) {
    console.error(`❌ Failed to read cache for ${url}:`, error);
    return null;
  }
}

/**
 * Write profile data to cache
 *
 * @param url - LinkedIn profile URL
 * @param profile - LinkedIn profile data to cache
 */
export function writeToCache(url: string, profile: any): void {
  ensureCacheDirectory();

  const filePath = getCachePath(url);
  if (!filePath) {
    console.error('❌ Invalid LinkedIn URL for cache write:', url);
    return;
  }

  const cacheData: CachedProfile = {
    url,
    cachedAt: new Date().toISOString(),
    profile,
  };

  try {
    fs.writeFileSync(filePath, JSON.stringify(cacheData, null, 2), 'utf-8');
    console.log(`💾 Cached: ${getCacheKey(url)}.json`);
  } catch (error) {
    console.error(`❌ Failed to write cache for ${url}:`, error);
  }
}

/**
 * Ensure cache directory exists
 * Creates directory if it doesn't exist
 */
export function ensureCacheDirectory(): void {
  if (!fs.existsSync(CACHE_DIR)) {
    fs.mkdirSync(CACHE_DIR, { recursive: true });
    console.log(`📁 Created cache directory: ${CACHE_DIR}`);
  }
}

/**
 * Check if cache is valid based on age
 *
 * @param cachedAt - ISO timestamp when cache was created
 * @param maxAgeDays - Maximum age in days
 * @returns true if cache is still valid
 */
export function isCacheValid(cachedAt: string, maxAgeDays: number): boolean {
  try {
    const cacheDate = new Date(cachedAt);
    const now = new Date();
    const diffDays = (now.getTime() - cacheDate.getTime()) / (1000 * 60 * 60 * 24);
    return diffDays <= maxAgeDays;
  } catch (error) {
    console.error('Error validating cache timestamp:', error);
    return false;
  }
}

/**
 * Clear all cached LinkedIn profiles
 * Useful for testing or forcing fresh scrapes
 */
export function clearCache(): void {
  if (fs.existsSync(CACHE_DIR)) {
    const files = fs.readdirSync(CACHE_DIR);
    let count = 0;

    files.forEach(file => {
      if (file.endsWith('.json')) {
        fs.unlinkSync(path.join(CACHE_DIR, file));
        count++;
      }
    });

    console.log(`🗑️  Cleared ${count} cached profiles`);
  } else {
    console.log('⚠️  Cache directory does not exist');
  }
}

/**
 * Get cache statistics
 * @returns Object with cache file count and total size
 */
export function getCacheStats(): { count: number; sizeKB: number } {
  if (!fs.existsSync(CACHE_DIR)) {
    return { count: 0, sizeKB: 0 };
  }

  const files = fs.readdirSync(CACHE_DIR);
  let totalSize = 0;
  let count = 0;

  files.forEach(file => {
    if (file.endsWith('.json')) {
      const filePath = path.join(CACHE_DIR, file);
      const stats = fs.statSync(filePath);
      totalSize += stats.size;
      count++;
    }
  });

  return {
    count,
    sizeKB: Math.round(totalSize / 1024),
  };
}
