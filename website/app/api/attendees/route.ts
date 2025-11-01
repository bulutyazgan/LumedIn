import { NextRequest, NextResponse } from 'next/server';
import { scrapeMultipleProfiles, LinkedInProfile, ScrapedResult } from '@/lib/linkedinScraper';

// Extended attendee interface with LinkedIn data
interface EnrichedAttendee {
  name: string;
  profileUrl: string;
  eventsAttended: number;
  instagram: string;
  x: string;
  tiktok: string;
  linkedin: string;
  website: string;

  // LinkedIn enrichment data
  linkedinData: LinkedInProfile | null;
  scrapingStatus: 'pending' | 'completed' | 'failed' | 'no_linkedin';
  scrapingError?: string;
}

interface StoredData {
  attendees: EnrichedAttendee[];
  eventUrl: string;
  timestamp: string;
  count: number;
  scrapingProgress: {
    total: number;
    completed: number;
    pending: number;
    failed: number;
  };
}

// In-memory storage for attendee data
let storedAttendees: StoredData = {
  attendees: [],
  eventUrl: '',
  timestamp: '',
  count: 0,
  scrapingProgress: {
    total: 0,
    completed: 0,
    pending: 0,
    failed: 0,
  },
};

// CORS headers
const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type',
};

export async function OPTIONS() {
  return NextResponse.json({}, { headers: corsHeaders });
}

export async function POST(request: NextRequest) {
  try {
    const data = await request.json();

    // Validate that we received attendee data
    if (!data.attendees || !Array.isArray(data.attendees)) {
      return NextResponse.json(
        { error: 'Invalid data format. Expected { attendees: [...] }' },
        { status: 400, headers: corsHeaders }
      );
    }

    // Transform attendees to enriched format with LinkedIn placeholders
    const enrichedAttendees: EnrichedAttendee[] = data.attendees.map((attendee: any) => ({
      ...attendee,
      linkedinData: null,
      scrapingStatus: attendee.linkedin ? 'pending' : 'no_linkedin',
    }));

    // Count how many have LinkedIn URLs
    const linkedinCount = enrichedAttendees.filter(a => a.linkedin).length;

    // Store the attendees with metadata
    storedAttendees = {
      attendees: enrichedAttendees,
      eventUrl: data.eventUrl || '',
      timestamp: new Date().toISOString(),
      count: enrichedAttendees.length,
      scrapingProgress: {
        total: linkedinCount,
        completed: 0,
        pending: linkedinCount,
        failed: 0,
      },
    };

    console.log(`Received ${data.attendees.length} attendees from ${data.eventUrl || 'unknown event'}`);
    console.log(`Starting LinkedIn scraping for ${linkedinCount} profiles...`);

    // Start background LinkedIn scraping (don't await)
    if (linkedinCount > 0) {
      startLinkedInScraping().catch(error => {
        console.error('Background LinkedIn scraping failed:', error);
      });
    }

    return NextResponse.json({
      success: true,
      message: `Successfully stored ${data.attendees.length} attendees`,
      count: data.attendees.length,
      linkedinScrapingStarted: linkedinCount > 0,
      linkedinProfilesQueued: linkedinCount,
    }, { headers: corsHeaders });

  } catch (error) {
    console.error('Error processing attendee data:', error);
    return NextResponse.json(
      { error: 'Failed to process attendee data' },
      { status: 500, headers: corsHeaders }
    );
  }
}

export async function GET() {
  // Return stored attendees for the dashboard
  return NextResponse.json(storedAttendees, { headers: corsHeaders });
}

/**
 * Background function to scrape LinkedIn profiles
 */
async function startLinkedInScraping() {
  const apiKey = process.env.SCRAPINGDOG_API_KEY;

  if (!apiKey) {
    console.error('SCRAPINGDOG_API_KEY not found in environment variables');
    // Mark all as failed
    storedAttendees.attendees.forEach(attendee => {
      if (attendee.scrapingStatus === 'pending') {
        attendee.scrapingStatus = 'failed';
        attendee.scrapingError = 'API key not configured';
      }
    });
    storedAttendees.scrapingProgress.failed = storedAttendees.scrapingProgress.total;
    storedAttendees.scrapingProgress.pending = 0;
    return;
  }

  // Extract LinkedIn URLs with their indices
  const linkedinUrls: Array<{ url: string; index: number }> = [];
  storedAttendees.attendees.forEach((attendee, index) => {
    if (attendee.linkedin && attendee.scrapingStatus === 'pending') {
      linkedinUrls.push({ url: attendee.linkedin, index });
    }
  });

  if (linkedinUrls.length === 0) {
    console.log('No LinkedIn URLs to scrape');
    return;
  }

  // Scrape profiles with progress tracking
  await scrapeMultipleProfiles(
    linkedinUrls.map(item => item.url),
    apiKey,
    (completed, total, lastResult: ScrapedResult) => {
      // Find the attendee index for this URL
      const urlItem = linkedinUrls.find(item => item.url === lastResult.url);
      if (!urlItem) return;

      const attendee = storedAttendees.attendees[urlItem.index];

      if (lastResult.success && lastResult.data) {
        // Successfully scraped - update with LinkedIn data
        attendee.linkedinData = lastResult.data;
        attendee.scrapingStatus = 'completed';
        storedAttendees.scrapingProgress.completed++;
      } else {
        // Failed - mark as failed with error
        attendee.scrapingStatus = 'failed';
        attendee.scrapingError = lastResult.error || 'Unknown error';
        storedAttendees.scrapingProgress.failed++;
      }

      storedAttendees.scrapingProgress.pending--;

      console.log(
        `LinkedIn scraping progress: ${completed}/${total} ` +
        `(✓ ${storedAttendees.scrapingProgress.completed} / ✗ ${storedAttendees.scrapingProgress.failed})`
      );
    }
  );

  console.log('LinkedIn scraping completed!');
  console.log(`Results: ${storedAttendees.scrapingProgress.completed} successful, ${storedAttendees.scrapingProgress.failed} failed`);
}
