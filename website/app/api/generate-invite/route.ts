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

    if (candidates.length === 0) {
      return NextResponse.json(
        { success: false, error: 'Must select at least 1 candidate' },
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
      // Team mode requires 3-5 candidates, but for 2 candidates we'll use individual mode
      if (candidates.length < 3) {
        return NextResponse.json(
          { success: false, error: 'Team invites require at least 3 candidates. For 1-2 candidates, individual mode will be used automatically.' },
          { status: 400 }
        );
      }

      if (candidates.length > 5) {
        return NextResponse.json(
          { success: false, error: 'Team invites support maximum 5 candidates' },
          { status: 400 }
        );
      }

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
      // Individual mode - works for any number of candidates
      const resultsMap = await generateIndividualInvitationEmails(
        candidates as CandidateProfile[],
        hackathonDescription
      );

      // Return the first candidate's email (when only 1 selected, this is the only one)
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
        { success: false, error: 'Failed to generate individual email' },
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
