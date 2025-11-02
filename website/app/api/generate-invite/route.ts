import { NextRequest, NextResponse } from 'next/server';
import {
  generateTeamInvitationEmail,
  generateIndividualInvitationEmails,
  CandidateProfile
} from '@/lib/openaiEmailGenerator';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { candidates, hackathonDescription, emailMode } = body;

    // Validate input
    if (!candidates || !Array.isArray(candidates)) {
      return NextResponse.json(
        { success: false, error: 'Invalid candidates data' },
        { status: 400 }
      );
    }

    if (candidates.length < 3 || candidates.length > 5) {
      return NextResponse.json(
        { success: false, error: 'Must select between 3 and 5 candidates' },
        { status: 400 }
      );
    }

    if (!hackathonDescription || hackathonDescription.trim().length < 10) {
      return NextResponse.json(
        { success: false, error: 'Hackathon description must be at least 10 characters' },
        { status: 400 }
      );
    }

    if (!emailMode || !['team', 'individual'].includes(emailMode)) {
      return NextResponse.json(
        { success: false, error: 'Invalid email mode' },
        { status: 400 }
      );
    }

    // Generate email based on mode
    if (emailMode === 'team') {
      const result = await generateTeamInvitationEmail(
        candidates as CandidateProfile[],
        hackathonDescription
      );

      if (!result.success) {
        return NextResponse.json(
          { success: false, error: result.error || 'Failed to generate team email' },
          { status: 500 }
        );
      }

      return NextResponse.json({
        success: true,
        emailDraft: result.emailDraft
      });
    } else {
      // Individual mode
      const resultsMap = await generateIndividualInvitationEmails(
        candidates as CandidateProfile[],
        hackathonDescription
      );

      // For now, return the first successful email or error
      // In the future, could return all emails
      for (const [candidateId, result] of resultsMap.entries()) {
        if (result.success && result.emailDraft) {
          return NextResponse.json({
            success: true,
            emailDraft: result.emailDraft,
            candidateId
          });
        }
      }

      return NextResponse.json(
        { success: false, error: 'Failed to generate individual emails' },
        { status: 500 }
      );
    }
  } catch (error) {
    console.error('Email generation API error:', error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : 'Internal server error'
      },
      { status: 500 }
    );
  }
}
