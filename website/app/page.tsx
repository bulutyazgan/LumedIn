'use client';

import { useEffect, useState } from 'react';
import styles from './page.module.css';

interface LinkedInProfile {
  profile_photo?: string;
  headline?: string;
  about?: string;
  experience?: Array<{
    position: string;
    company_name: string;
    location: string;
    starts_at: string;
    ends_at: string;
    duration: string;
  }>;
  education?: Array<{
    college_name: string;
    college_degree: string;
    college_degree_field: string | null;
    college_duration: string;
  }>;
  articles?: Array<any>;
  description?: {
    description1?: string;
    description2?: string;
    description3?: string;
  };
  activities?: Array<{
    link: string;
    title: string;
    activity: string;
  }>;
  certification?: Array<{
    certification: string;
    company_name: string;
    issue_date: string;
  }>;
}

interface Attendee {
  name: string;
  profileUrl: string;
  eventsAttended?: number;
  instagram?: string;
  x?: string;
  tiktok?: string;
  linkedin?: string;
  website?: string;

  // LinkedIn enrichment
  linkedinData?: LinkedInProfile | null;
  scrapingStatus?: 'pending' | 'completed' | 'failed' | 'no_linkedin';
  scrapingError?: string;
}

interface AttendeeData {
  attendees: Attendee[];
  eventUrl?: string;
  timestamp?: string;
  count?: number;
  scrapingProgress?: {
    total: number;
    completed: number;
    pending: number;
    failed: number;
  };
}

export default function Dashboard() {
  const [data, setData] = useState<AttendeeData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expandedRows, setExpandedRows] = useState<Set<number>>(new Set());

  const fetchData = async (preserveScrollPosition = false) => {
    try {
      if (!preserveScrollPosition) {
        setLoading(true);
      }
      const response = await fetch('/api/attendees');
      if (!response.ok) {
        throw new Error('Failed to fetch attendee data');
      }
      const result = await response.json();
      setData(result);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      if (!preserveScrollPosition) {
        setLoading(false);
      }
    }
  };

  useEffect(() => {
    fetchData();
    // Auto-refresh every 5 seconds to catch LinkedIn scraping progress
    const interval = setInterval(() => fetchData(true), 5000);
    return () => clearInterval(interval);
  }, []);

  const toggleRow = (index: number) => {
    const newExpanded = new Set(expandedRows);
    if (newExpanded.has(index)) {
      newExpanded.delete(index);
    } else {
      newExpanded.add(index);
    }
    setExpandedRows(newExpanded);
  };

  const downloadCSV = () => {
    if (!data || !data.attendees || data.attendees.length === 0) return;

    const rows = [
      ['Name', 'Profile URL', 'Events Attended', 'Instagram', 'X', 'TikTok', 'LinkedIn', 'Website', 'Headline', 'About', 'Scraping Status']
    ];

    for (const attendee of data.attendees) {
      rows.push([
        attendee.name,
        attendee.profileUrl,
        String(attendee.eventsAttended || 0),
        attendee.instagram || '',
        attendee.x || '',
        attendee.tiktok || '',
        attendee.linkedin || '',
        attendee.website || '',
        attendee.linkedinData?.headline || '',
        attendee.linkedinData?.about || '',
        attendee.scrapingStatus || ''
      ]);
    }

    const csv = rows.map(r => r.map(x => `"${(x||'').replace(/"/g, '""')}"`).join(',')).join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `luma_attendees_enriched_${new Date().toISOString().split('T')[0]}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  if (loading) {
    return (
      <div className={styles.container}>
        <div className={styles.loading}>Loading attendee data...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className={styles.container}>
        <div className={styles.error}>Error: {error}</div>
      </div>
    );
  }

  const attendees = data?.attendees || [];
  const hasData = attendees.length > 0;
  const progress = data?.scrapingProgress;

  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <h1>Luma Attendee Dashboard</h1>
        {data?.eventUrl && (
          <p className={styles.eventUrl}>
            Event: <a href={data.eventUrl} target="_blank" rel="noopener noreferrer">{data.eventUrl}</a>
          </p>
        )}
        {data?.timestamp && (
          <p className={styles.timestamp}>Last updated: {new Date(data.timestamp).toLocaleString()}</p>
        )}
      </header>

      {/* LinkedIn Scraping Progress */}
      {progress && progress.total > 0 && (
        <div className={styles.progressSection}>
          <h3>LinkedIn Scraping Progress</h3>
          <div className={styles.progressBar}>
            <div
              className={styles.progressFill}
              style={{
                width: `${(progress.completed / progress.total) * 100}%`,
                backgroundColor: progress.pending > 0 ? '#4caf50' : '#2196f3'
              }}
            />
          </div>
          <div className={styles.progressStats}>
            <span className={styles.progressCompleted}>✓ {progress.completed} completed</span>
            <span className={styles.progressPending}>⏳ {progress.pending} pending</span>
            <span className={styles.progressFailed}>✗ {progress.failed} failed</span>
            <span className={styles.progressTotal}>Total: {progress.total}</span>
          </div>
        </div>
      )}

      {!hasData ? (
        <div className={styles.noData}>
          <h2>No attendee data yet</h2>
          <p>Use the Chrome extension on a Luma event page to analyze attendees.</p>
          <p>The data will appear here automatically.</p>
        </div>
      ) : (
        <>
          <div className={styles.summary}>
            <h2>Summary</h2>
            <p>Total Attendees: <strong>{attendees.length}</strong></p>
            <button onClick={downloadCSV} className={styles.downloadBtn}>
              Download CSV
            </button>
          </div>

          <div className={styles.tableContainer}>
            <table className={styles.table}>
              <thead>
                <tr>
                  <th>Expand</th>
                  <th>Photo</th>
                  <th>Name</th>
                  <th>Headline</th>
                  <th>Events</th>
                  <th>LinkedIn</th>
                  <th>Status</th>
                  <th>Socials</th>
                </tr>
              </thead>
              <tbody>
                {attendees.map((attendee, index) => (
                  <>
                    <tr key={index} className={expandedRows.has(index) ? styles.expanded : ''}>
                      <td>
                        <button
                          onClick={() => toggleRow(index)}
                          className={styles.expandBtn}
                        >
                          {expandedRows.has(index) ? '▼' : '▶'}
                        </button>
                      </td>
                      <td>
                        {attendee.linkedinData?.profile_photo ? (
                          <img
                            src={attendee.linkedinData.profile_photo}
                            alt={attendee.name}
                            className={styles.profilePhoto}
                          />
                        ) : (
                          <div className={styles.noPhoto}>👤</div>
                        )}
                      </td>
                      <td>
                        <strong>{attendee.name}</strong>
                        <br />
                        <a href={attendee.profileUrl} target="_blank" rel="noopener noreferrer" className={styles.smallLink}>
                          Luma Profile
                        </a>
                      </td>
                      <td className={styles.headline}>
                        {attendee.linkedinData?.headline || (
                          attendee.scrapingStatus === 'pending' ? (
                            <span className={styles.loading}>Loading...</span>
                          ) : '-'
                        )}
                      </td>
                      <td>{attendee.eventsAttended || 0}</td>
                      <td>
                        {attendee.linkedin ? (
                          <a href={attendee.linkedin} target="_blank" rel="noopener noreferrer">
                            View
                          </a>
                        ) : '-'}
                      </td>
                      <td>
                        <span className={`${styles.status} ${styles[attendee.scrapingStatus || 'no_linkedin']}`}>
                          {attendee.scrapingStatus === 'pending' && '⏳ Scraping...'}
                          {attendee.scrapingStatus === 'completed' && '✓ Done'}
                          {attendee.scrapingStatus === 'failed' && '✗ Failed'}
                          {attendee.scrapingStatus === 'no_linkedin' && '-'}
                        </span>
                        {attendee.scrapingError && (
                          <div className={styles.error}>{attendee.scrapingError}</div>
                        )}
                      </td>
                      <td className={styles.socials}>
                        {attendee.instagram && <a href={attendee.instagram} target="_blank" rel="noopener noreferrer">IG</a>}
                        {attendee.x && <a href={attendee.x} target="_blank" rel="noopener noreferrer">X</a>}
                        {attendee.tiktok && <a href={attendee.tiktok} target="_blank" rel="noopener noreferrer">TT</a>}
                        {attendee.website && <a href={attendee.website} target="_blank" rel="noopener noreferrer">Web</a>}
                      </td>
                    </tr>

                    {/* Expanded Row Details */}
                    {expandedRows.has(index) && attendee.linkedinData && (
                      <tr key={`${index}-expanded`} className={styles.expandedContent}>
                        <td colSpan={8}>
                          <div className={styles.detailsContainer}>
                            {/* About */}
                            {attendee.linkedinData.about && (
                              <div className={styles.detailSection}>
                                <h4>About</h4>
                                <p>{attendee.linkedinData.about}</p>
                              </div>
                            )}

                            {/* Experience */}
                            {attendee.linkedinData.experience && attendee.linkedinData.experience.length > 0 && (
                              <div className={styles.detailSection}>
                                <h4>Experience ({attendee.linkedinData.experience.length})</h4>
                                {attendee.linkedinData.experience.slice(0, 3).map((exp, i) => (
                                  <div key={i} className={styles.experienceItem}>
                                    <strong>{exp.position}</strong> at {exp.company_name}
                                    <br />
                                    <span className={styles.meta}>{exp.duration} • {exp.location}</span>
                                  </div>
                                ))}
                                {attendee.linkedinData.experience.length > 3 && (
                                  <p className={styles.more}>+ {attendee.linkedinData.experience.length - 3} more</p>
                                )}
                              </div>
                            )}

                            {/* Education */}
                            {attendee.linkedinData.education && attendee.linkedinData.education.length > 0 && (
                              <div className={styles.detailSection}>
                                <h4>Education</h4>
                                {attendee.linkedinData.education.map((edu, i) => (
                                  <div key={i} className={styles.educationItem}>
                                    <strong>{edu.college_name}</strong>
                                    {edu.college_degree && <span> - {edu.college_degree}</span>}
                                    {edu.college_degree_field && <span> ({edu.college_degree_field})</span>}
                                    <br />
                                    <span className={styles.meta}>{edu.college_duration}</span>
                                  </div>
                                ))}
                              </div>
                            )}

                            {/* Certifications */}
                            {attendee.linkedinData.certification && attendee.linkedinData.certification.length > 0 && (
                              <div className={styles.detailSection}>
                                <h4>Certifications ({attendee.linkedinData.certification.length})</h4>
                                {attendee.linkedinData.certification.slice(0, 3).map((cert, i) => (
                                  <div key={i} className={styles.certificationItem}>
                                    <strong>{cert.certification}</strong>
                                    <br />
                                    <span className={styles.meta}>{cert.company_name} • {cert.issue_date}</span>
                                  </div>
                                ))}
                                {attendee.linkedinData.certification.length > 3 && (
                                  <p className={styles.more}>+ {attendee.linkedinData.certification.length - 3} more</p>
                                )}
                              </div>
                            )}

                            {/* Activities */}
                            {attendee.linkedinData.activities && attendee.linkedinData.activities.length > 0 && (
                              <div className={styles.detailSection}>
                                <h4>Recent Activity ({attendee.linkedinData.activities.length})</h4>
                                {attendee.linkedinData.activities.slice(0, 3).map((activity, i) => (
                                  <div key={i} className={styles.activityItem}>
                                    <a href={activity.link} target="_blank" rel="noopener noreferrer">
                                      {activity.title.substring(0, 100)}{activity.title.length > 100 ? '...' : ''}
                                    </a>
                                    <br />
                                    <span className={styles.meta}>{activity.activity}</span>
                                  </div>
                                ))}
                                {attendee.linkedinData.activities.length > 3 && (
                                  <p className={styles.more}>+ {attendee.linkedinData.activities.length - 3} more activities</p>
                                )}
                              </div>
                            )}
                          </div>
                        </td>
                      </tr>
                    )}
                  </>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}
