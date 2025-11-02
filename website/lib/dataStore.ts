// Shared in-memory data store for attendees
// This allows multiple API routes to access the same data
// Uses global singleton pattern to survive Next.js HMR (Hot Module Replacement)

// Extend global namespace to include our dataStore
declare global {
  var __dataStore: DataStore | undefined;
}

interface LinkedInProfile {
  profile_photo?: string;
  headline?: string;
  about?: string;
  location?: string;
  connections?: string;
  experience?: Array<any>;
  education?: Array<any>;
  certification?: Array<any>;
  activities?: Array<any>;
}

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

  // OpenAI scoring data
  hackathons_won: number | string | null;
  overall_score: number | null;
  technical_skill_summary: string | null;
  collaboration_summary: string | null;
  summary: string | null;
  scoringStatus: 'pending' | 'completed' | 'failed' | 'skipped';
  scoringError?: string;
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
  scoringProgress: {
    total: number;
    completed: number;
    pending: number;
    failed: number;
    skipped: number;
  };
}

// Singleton data store
class DataStore {
  private static instance: DataStore;
  private data: StoredData;
  public isScrapingInProgress: boolean = false;
  public isScoringInProgress: boolean = false;

  private constructor() {
    this.data = {
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
      scoringProgress: {
        total: 0,
        completed: 0,
        pending: 0,
        failed: 0,
        skipped: 0,
      },
    };
  }

  public static getInstance(): DataStore {
    // Use global object to survive Next.js HMR in development mode
    // This prevents multiple instances from being created when modules hot-reload
    if (!global.__dataStore) {
      global.__dataStore = new DataStore();
      console.log('📦 DataStore: New instance created');
    } else {
      console.log('📦 DataStore: Using existing instance');
    }
    return global.__dataStore;
  }

  public getData(): StoredData {
    return this.data;
  }

  public setData(data: StoredData): void {
    this.data = data;
    console.log(`📦 DataStore: Data updated - ${data.attendees.length} attendees stored`);
  }

  public getAttendees(): EnrichedAttendee[] {
    return this.data.attendees;
  }

  public setAttendees(attendees: EnrichedAttendee[]): void {
    this.data.attendees = attendees;
  }

  public updateAttendee(index: number, updates: Partial<EnrichedAttendee>): void {
    if (index >= 0 && index < this.data.attendees.length) {
      this.data.attendees[index] = {
        ...this.data.attendees[index],
        ...updates,
      };
    }
  }

  public clear(): void {
    this.data = {
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
      scoringProgress: {
        total: 0,
        completed: 0,
        pending: 0,
        failed: 0,
        skipped: 0,
      },
    };
    this.isScrapingInProgress = false;
    this.isScoringInProgress = false;
  }
}

// Export singleton instance
export const dataStore = DataStore.getInstance();
export type { StoredData, EnrichedAttendee, LinkedInProfile };
