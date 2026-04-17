import React, { useState, useEffect, useMemo } from 'react';
import './App.css';
import type { MatchData, GateParams } from './types';

function App() {
  const [model, setModel] = useState("v13");
  const [data, setData] = useState<MatchData[]>([]);
  const [loading, setLoading] = useState(true);

  // Sorting
  const [sortField, setSortField] = useState<keyof MatchData | 'earned'>('date');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');
  
  // Navigation
  const [currentView, setCurrentView] = useState<'matches' | 'leagues'>('matches');
  const [sidebarOpen, setSidebarOpen] = useState(true);
  
  // Expanded row
  const [expandedMatch, setExpandedMatch] = useState<string | null>(null);

  const [params, setParams] = useState<GateParams>({
    minConfidenceQ3: 0.62,
    minConfidenceQ4: 0.55,
    maxVolatility: 0.70,
    allowWomen: false,
    odds: 1.4,
    kellyFraction: 0.25,
    bankroll: 1000,
    startDate: "2026-04-01",
    endDate: "2026-04-15"
  });

  useEffect(() => {
    setLoading(true);
    fetch(`/dashboard_data_${model}.json`)
      .then(res => {
         if (!res.ok) throw new Error("Not found");
         return res.json();
      })
      .then((json: MatchData[]) => {
        setData(json);
        setLoading(false);
      })
      .catch(err => {
        // Fallback to original dashboard_data.json if v13 is selected and v13 file doesnt exist by name
        if (model === 'v13') {
           fetch(`/dashboard_data.json`).then(r=>{if(!r.ok) throw new Error(); return r.json()}).then(j=>{setData(j); setLoading(false);}).catch(()=>{setLoading(false)});
           return;
        }
        console.error("Failed to load data", err);
        setLoading(false);
        setData([]);
      });
  }, [model]);

  // Determine dynamic limits based on current data
  const dataLimits = useMemo(() => {
    if (!data.length) return { minQ3: 0.5, maxQ3: 1, minQ4: 0.5, maxQ4: 1 };
    const q3Conf = data.filter(d => d.target==='q3').map(d => d.confidence);
    const q4Conf = data.filter(d => d.target==='q4').map(d => d.confidence);
    
    return {
      minQ3: q3Conf.length ? Math.max(0, Math.min(...q3Conf)) : 0,
      maxQ3: q3Conf.length ? Math.min(1, Math.max(...q3Conf)) : 1,
      minQ4: q4Conf.length ? Math.max(0, Math.min(...q4Conf)) : 0,
      maxQ4: q4Conf.length ? Math.min(1, Math.max(...q4Conf)) : 1,
    };
  }, [data]);

  // Adjust parameters if out of bounds after switching dataset
  useEffect(() => {
    setParams(prev => ({
      ...prev,
      minConfidenceQ3: Math.max(dataLimits.minQ3, Math.min(prev.minConfidenceQ3, dataLimits.maxQ3)),
      minConfidenceQ4: Math.max(dataLimits.minQ4, Math.min(prev.minConfidenceQ4, dataLimits.maxQ4))
    }));
  }, [dataLimits]);

  const handleChange = (key: keyof GateParams, value: number | boolean | string) => {
    setParams(prev => ({ ...prev, [key]: value }));
  };

  const sortBy = (field: keyof MatchData | 'earned') => {
    if (sortField === field) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortOrder('desc');
    }
  };

  const exportConfig = () => {
    const config = {
      model_version: model,
      gates: {
        min_confidence_q3: params.minConfidenceQ3,
        min_confidence_q4: params.minConfidenceQ4,
        max_volatility: params.maxVolatility,
        allow_women: params.allowWomen,
        date_range: { start: params.startDate, end: params.endDate }
      },
      betting: {
        odds: params.odds,
        kelly_fraction: params.kellyFraction
      }
    };
    
    const blob = new Blob([JSON.stringify(config, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${model}_gates_config.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Filter and calculate results
  const results = useMemo(() => {
    let betsCount = 0;
    let hits = 0;
    let bankroll = params.bankroll;
    
    const p_win = 0.55; 
    const b = params.odds - 1;
    const q = 1 - p_win;
    const kelly = (b * p_win - q) / b;
    
    const filterStart = new Date(params.startDate).getTime();
    const filterEnd = new Date(params.endDate).getTime();
    
    const filteredMatches = data.filter(match => {
      const matchTime = new Date(match.date).getTime();
      if (matchTime < filterStart || matchTime > filterEnd + 86400000) return false;
      if (!params.allowWomen && match.gender === "women") return false;
      if (match.volatility > params.maxVolatility) return false;
      if (match.target === "q3" && match.confidence < params.minConfidenceQ3) return false;
      if (match.target === "q4" && match.confidence < params.minConfidenceQ4) return false;
      return true;
    });

    const betLog: (MatchData & { stake: number, earned: number, newBankroll: number })[] = [];

    // Chronological simulation for ROI calculation
    const chronoMatches = [...filteredMatches].sort((a,b) => new Date(a.date).getTime() - new Date(b.date).getTime());
    
    // League Aggregation
    const leagueStats: Record<string, { bets: number, hits: number, bankroll: number, roi: number, earned: number }> = {};
    chronoMatches.forEach(match => {
      if (!leagueStats[match.league]) {
        leagueStats[match.league] = { bets: 0, hits: 0, bankroll: params.bankroll, roi: 0, earned: 0 };
      }
    });

    chronoMatches.forEach(match => {
      betsCount++;
      let stake = bankroll * (kelly > 0 ? kelly : 0.05) * params.kellyFraction;
      stake = Math.min(stake, bankroll * 0.05); // cap at 5%
      if (stake < 1) stake = 1;

      // League specific stake
      let lStake = leagueStats[match.league].bankroll * (kelly > 0 ? kelly : 0.05) * params.kellyFraction;
      lStake = Math.min(lStake, leagueStats[match.league].bankroll * 0.05);
      if (lStake < 1) lStake = 1;

      let earned = -stake;
      let lEarned = -lStake;
      
      if (match.hit) {
        hits++;
        earned = stake * (params.odds - 1);
        lEarned = lStake * (params.odds - 1);
        leagueStats[match.league].hits++;
      }
      
      bankroll += earned;
      leagueStats[match.league].bankroll += lEarned;
      leagueStats[match.league].bets++;
      leagueStats[match.league].earned += lEarned;

      betLog.push({ ...match, stake, earned, newBankroll: bankroll });
    });

    Object.keys(leagueStats).forEach(l => {
        leagueStats[l].roi = ((leagueStats[l].bankroll - params.bankroll) / params.bankroll) * 100;
    });
    
    // Sort league stats by ROI desc
    const sortedLeagues = Object.entries(leagueStats).sort((a, b) => b[1].roi - a[1].roi);

    // Apply Sorting
    betLog.sort((a, b) => {
      let valA: any = a[sortField];
      let valB: any = b[sortField];
      
      if (sortField === 'date') {
        valA = new Date(valA).getTime();
        valB = new Date(valB).getTime();
      }
      if (valA < valB) return sortOrder === 'asc' ? -1 : 1;
      if (valA > valB) return sortOrder === 'asc' ? 1 : -1;
      return 0;
    });

    const hitRate = betsCount > 0 ? (hits / betsCount) * 100 : 0;
    const roi = ((bankroll - params.bankroll) / params.bankroll) * 100;

    return { betsCount, hits, hitRate, bankroll, roi, betLog, sortedLeagues };
  }, [data, params, sortField, sortOrder]);


  return (
    <div className="app-container">
      {/* Floating Menu Button */}
      {!sidebarOpen && (
        <button className="icon-btn glass-panel" onClick={() => setSidebarOpen(true)} style={{ position: 'absolute', top: 16, left: 16, zIndex: 100, background: 'var(--panel-bg)', padding: 12 }}>
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="3" y1="12" x2="21" y2="12"></line><line x1="3" y1="6" x2="21" y2="6"></line><line x1="3" y1="18" x2="21" y2="18"></line></svg>
        </button>
      )}

      <div className="dashboard-layout" style={{ display: 'flex', overflow: 'hidden', flex: 1 }}>
        {/* Sidebar */}
        <aside className={`glass-panel controls-panel sidebar ${sidebarOpen ? 'open' : 'closed'}`} style={{ zIndex: 20 }}>
          
          {/* Header Controls inside Sidebar */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
             <h1 className="text-gradient" style={{ margin: 0, fontSize: '1.8rem' }}>Pulpa</h1>
             <button className="icon-btn" onClick={() => setSidebarOpen(false)}>
               <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
             </button>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 12, marginBottom: 24 }}>
            <select 
              className="select-input" 
              value={model} 
              onChange={e => setModel(e.target.value)}
            >
              <option value="v13">Modelo V13</option>
              <option value="v15">Modelo V15 (Nuevo)</option>
            </select>
            <button className="export-btn" onClick={exportConfig} style={{ justifyContent: 'center' }}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>
              Export Configuration
            </button>
          </div>

          <div className="sidebar-nav" style={{ marginBottom: 24 }}>
            <button 
              className={`sidebar-link ${currentView === 'matches' ? 'active' : ''}`}
              onClick={() => setCurrentView('matches')}
            >
              📊 Match History
            </button>
            <button 
              className={`sidebar-link ${currentView === 'leagues' ? 'active' : ''}`}
              onClick={() => setCurrentView('leagues')}
            >
              🏆 League Summary
            </button>
          </div>

          <hr style={{ borderColor: 'var(--border-color)', width: '100%', margin: '0 0 16px 0' }} />

          <h3 style={{ margin: '0 0 12px 0' }}>Data Filters</h3>
          
          <div className="control-group">
            <label>Start Date</label>
            <input 
              type="date"
              className="select-input"
              value={params.startDate}
              onChange={e => handleChange('startDate', e.target.value)} 
            />
          </div>
          <div className="control-group">
            <label>End Date</label>
            <input 
              type="date"
              className="select-input"
              value={params.endDate}
              onChange={e => handleChange('endDate', e.target.value)} 
            />
          </div>
          
          <div className="control-group">
            <label>
              Min Confidence Q3
              <span className="value-badge">{params.minConfidenceQ3.toFixed(2)}</span>
            </label>
            <input 
              type="range" min={dataLimits.minQ3.toFixed(2)} max={dataLimits.maxQ3.toFixed(2)} step="0.01" 
              value={params.minConfidenceQ3} 
              onChange={e => handleChange('minConfidenceQ3', parseFloat(e.target.value))} 
            />
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7em', color: '#555' }}>
              <span>{dataLimits.minQ3.toFixed(2)}</span><span>{dataLimits.maxQ3.toFixed(2)}</span>
            </div>
          </div>
          
          <div className="control-group">
            <label>
              Min Confidence Q4
              <span className="value-badge">{params.minConfidenceQ4.toFixed(2)}</span>
            </label>
            <input 
              type="range" min={dataLimits.minQ4.toFixed(2)} max={dataLimits.maxQ4.toFixed(2)} step="0.01" 
              value={params.minConfidenceQ4} 
              onChange={e => handleChange('minConfidenceQ4', parseFloat(e.target.value))} 
            />
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7em', color: '#555' }}>
              <span>{dataLimits.minQ4.toFixed(2)}</span><span>{dataLimits.maxQ4.toFixed(2)}</span>
            </div>
          </div>

          <div className="control-group">
            <label>
              Max Volatility
              <span className="value-badge">{params.maxVolatility.toFixed(2)}</span>
            </label>
            <input 
              type="range" min="0.1" max="1.0" step="0.05" 
              value={params.maxVolatility} 
              onChange={e => handleChange('maxVolatility', parseFloat(e.target.value))} 
            />
          </div>

          <div className="toggle-group" style={{ marginTop: '8px' }}>
            <span style={{ fontWeight: 600, fontSize: '0.9rem', color: 'var(--text-muted)' }}>Allow Women's Leagues</span>
            <input 
              type="checkbox" 
              checked={params.allowWomen} 
              onChange={e => handleChange('allowWomen', e.target.checked)} 
            />
          </div>

          <hr style={{ borderColor: 'var(--border-color)', width: '100%', margin: '16px 0' }} />
          <h3 style={{ margin: '0 0 8px 0' }}>Simulator Config</h3>

          <div className="control-group">
            <label>Odds <span className="value-badge">{params.odds.toFixed(2)}</span></label>
            <input 
              type="range" min="1.50" max="2.50" step="0.01" 
              value={params.odds} 
              onChange={e => handleChange('odds', parseFloat(e.target.value))} 
            />
          </div>
        </aside>

        {/* Main Content */}
        <main className="main-content" style={{ minWidth: 0, flex: 1, padding: 24, paddingLeft: sidebarOpen ? 24 : 48, overflowY: 'hidden', display: 'flex', flexDirection: 'column' }}>
          {/* Metrics */}
          <div className="metrics-grid animate-fade-in" style={{ animationDelay: '0.1s' }}>
            <div className={`glass-panel metric-card ${results.roi >= 0 ? 'positive' : 'negative'}`}>
              <span className="metric-label">Total ROI</span>
              <h2 className={`metric-value ${results.roi >= 0 ? 'positive' : 'negative'}`}>
                {results.roi >= 0 ? '+' : ''}{results.roi.toLocaleString('en-US', { minimumFractionDigits: 1, maximumFractionDigits: 1 })}%
              </h2>
            </div>
            <div className="glass-panel metric-card">
              <span className="metric-label">Hit Rate</span>
              <h2 className="metric-value">
                {results.hitRate.toLocaleString('en-US', { minimumFractionDigits: 1, maximumFractionDigits: 1 })}%
              </h2>
            </div>
            <div className="glass-panel metric-card">
              <span className="metric-label">Total Bets placed</span>
              <h2 className="metric-value">{results.betsCount.toLocaleString('en-US')}</h2>
            </div>
            <div className={`glass-panel metric-card ${results.bankroll >= params.bankroll ? 'positive' : 'negative'}`}>
              <span className="metric-label">Final Bankroll</span>
              <h2 className={`metric-value ${results.bankroll >= params.bankroll ? 'positive' : 'negative'}`}>
                ${results.bankroll.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
              </h2>
            </div>
          </div>



          {/* League Stats Table */}
          {currentView === 'leagues' && results.sortedLeagues.length > 0 && (
            <div className="glass-panel data-table-container animate-fade-in" style={{ animationDelay: '0.15s', marginBottom: '24px' }}>
              <h3 style={{ margin: '16px', color: 'var(--text-main)' }}>Resultados por Liga</h3>
              <table style={{ borderTop: '1px solid var(--border-color)' }}>
                <thead>
                  <tr>
                    <th>League</th>
                    <th>Bets</th>
                    <th>Hits</th>
                    <th>Hit Rate</th>
                    <th>P/L</th>
                    <th>ROI</th>
                  </tr>
                </thead>
                <tbody>
                  {results.sortedLeagues.map(([league, stats], i) => {
                    const hitRate = stats.bets > 0 ? (stats.hits / stats.bets) * 100 : 0;
                    return (
                      <tr key={i}>
                        <td style={{ fontWeight: 600 }}>{league}</td>
                        <td>{stats.bets}</td>
                        <td>{stats.hits}</td>
                        <td>{hitRate.toFixed(1)}%</td>
                        <td style={{ color: stats.earned > 0 ? 'var(--success)' : 'var(--danger)' }}>
                            {stats.earned > 0 ? '+' : ''}{stats.earned.toFixed(2)}
                        </td>
                        <td style={{ color: stats.roi > 0 ? 'var(--success)' : 'var(--danger)' }}>
                            {stats.roi.toFixed(1)}%
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}

          {/* Matches Table */}
          {currentView === 'matches' && (
            <div className="glass-panel data-table-container animate-fade-in" style={{ animationDelay: '0.2s' }}>
              {loading ? (
              <div style={{ padding: 40, textAlign: 'center' }}>Loading Model Data...</div>
            ) : (
            <table>
              <thead>
                <tr>
                  <th onClick={() => sortBy('date')} style={{ cursor: 'pointer' }}>Date {sortField === 'date' && (sortOrder === 'asc' ? '↑' : '↓')}</th>
                  <th onClick={() => sortBy('league')} style={{ cursor: 'pointer' }}>League {sortField === 'league' && (sortOrder === 'asc' ? '↑' : '↓')}</th>
                  <th onClick={() => sortBy('target')} style={{ cursor: 'pointer' }}>Target {sortField === 'target' && (sortOrder === 'asc' ? '↑' : '↓')}</th>
                  <th onClick={() => sortBy('winner_pick')} style={{ cursor: 'pointer' }}>Pick {sortField === 'winner_pick' && (sortOrder === 'asc' ? '↑' : '↓')}</th>
                  <th onClick={() => sortBy('confidence')} style={{ cursor: 'pointer' }}>Conf. {sortField === 'confidence' && (sortOrder === 'asc' ? '↑' : '↓')}</th>
                  <th>Result</th>
                  <th onClick={() => sortBy('earned')} style={{ cursor: 'pointer' }}>P/L {sortField === 'earned' && (sortOrder === 'asc' ? '↑' : '↓')}</th>
                </tr>
              </thead>
              <tbody>
                {results.betLog.slice(0, 100).map((bet, i) => (
                  <React.Fragment key={i}>
                    <tr onClick={() => setExpandedMatch(expandedMatch === bet.match_id ? null : bet.match_id)} style={{ cursor: 'pointer' }}>
                      <td>{bet.date.split('T')[0]}</td>
                      <td>{bet.league}</td>
                      <td><span className={`badge ${bet.target}`}>{bet.target.toUpperCase()}</span></td>
                      <td style={{ textTransform: 'capitalize' }}>{bet.winner_pick}</td>
                      <td style={{ fontWeight: 'bold' }}>{bet.confidence.toFixed(3)}</td>
                      <td>
                        <span className={`badge ${bet.hit ? 'win' : 'loss'}`}>
                          {bet.hit ? 'WIN' : 'LOSS'}
                        </span>
                      </td>
                      <td style={{ color: bet.earned > 0 ? 'var(--success)' : 'var(--danger)', fontWeight: 600 }}>
                        {bet.earned > 0 ? '+' : ''}{bet.earned.toFixed(2)}
                      </td>
                    </tr>
                    {expandedMatch === bet.match_id && (
                      <tr className="expanded-row">
                        <td colSpan={7}>
                          <div className="match-details">
                            <div className="match-info">
                              <p><strong>Match ID:</strong> <span style={{ fontFamily: 'monospace' }}>{bet.match_id}</span></p>
                              <p><strong>Gender:</strong> {bet.gender.toUpperCase()}</p>
                              <div style={{ display: 'flex', gap: '16px', marginTop: 12 }}>
                                <div>
                                  <p><strong>Q1:</strong> Home {bet.q1_home} - {bet.q1_away} Away</p>
                                  <p><strong>Q2:</strong> Home {bet.q2_home} - {bet.q2_away} Away</p>
                                </div>
                                {bet.target === 'q4' && (
                                  <div>
                                    <p><strong>Q3:</strong> Home {bet.q3_home} - {bet.q3_away} Away</p>
                                  </div>
                                )}
                              </div>
                            </div>
                            <a 
                              href={`https://www.sofascore.com/match/${bet.match_id}`} 
                              target="_blank" 
                              rel="noopener noreferrer" 
                              className="sofascore-btn"
                              onClick={(e) => e.stopPropagation()}
                            >
                              ⚽ Ver en SofaScore
                            </a>
                            <a 
                              href={`https://t.me/Bot?start=${bet.match_id}`} 
                              target="_blank" 
                              rel="noopener noreferrer" 
                              className="sofascore-btn"
                              style={{ background: '#0088cc', marginLeft: 8 }}
                              onClick={(e) => e.stopPropagation()}
                            >
                              🤖 Enviar al Bot Telegram
                            </a>
                          </div>
                        </td>
                      </tr>
                    )}
                  </React.Fragment>
                ))}
              </tbody>
            </table>
            )}
            {results.betLog.length > 100 && (
              <div style={{ padding: 16, textAlign: 'center', color: 'var(--text-muted)' }}>
                Showing last 100 of {results.betLog.length} bets...
              </div>
            )}
            {!loading && results.betLog.length === 0 && (
              <div style={{ padding: 40, textAlign: 'center', color: 'var(--text-muted)' }}>
                No bets qualify under current gate parameters.
              </div>
            )}
            </div>
          )}
        </main>
      </div>
    </div>
  );
}

export default App;
