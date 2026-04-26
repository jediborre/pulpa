import React, { useState, useEffect, useMemo, useCallback } from 'react';
import './App.css';
import type { MatchData, GateParams } from './types';

const VERSIONS = ['v2', 'v4', 'v6', 'v9', 'v12', 'v13', 'v15', 'v16', 'v17'];

interface ModelTechInfo {
  algo: string;
  features: string;
  samples: string;
  dates: string;
  timing: string;
  stats: string;
}

const MODEL_INFO: Record<string, ModelTechInfo> = {
  v2: { 
    algo: 'Logistic Regression Baseline', 
    features: 'Proxy de fuerza histórica. Segmentación inicial de equipos en buckets de volumen.',
    samples: '~10,000 partidos',
    dates: 'Histórico estático (Entrenado en bulk previo)',
    timing: 'Pre-match / Offline baseline no-dinámico.',
    stats: 'Hit Rate: ~58-60% | ROI Promedio: 1-2%' 
  },
  v4: { 
    algo: 'Random Forest + Momentum', 
    features: 'Simulaciones Monte Carlo generativas, métricas de remontada (puntos faltantes liderar), clutch points (ult. 6m).',
    samples: '~12,000+ partidos',
    dates: 'Histórico hasta inicios de 2026',
    timing: 'Pre-Start del cuarto (Offline testing base).',
    stats: 'Hit Rate: ~61-64% | ROI Promedio: ~3-4%' 
  },
  v6: { 
    algo: 'LightGBM Gradient Boosting', 
    features: 'Multi-Agent Monte Carlo, optimización iterativa de árboles, ranking probabilístico ajustado (ROC AUC).',
    samples: '~15,000 partidos',
    dates: 'Histórico Walk-Forward simulado inicial',
    timing: 'Simulado al inicio del cuarto (Offline / Experimental).',
    stats: 'Hit Rate: ~64-65% | ROI Promedio: ~5-6%' 
  },
  v9: { 
    algo: 'XGBoost Optimizado', 
    features: 'Ensembles Monte Carlo, control estricto de volatilidad y gates probabilísticos. Focus en reducir varianza.',
    samples: '~15,000+ partidos',
    dates: 'Walk-Forward temprano',
    timing: 'Live Emulado (puntos fijos de evaluación).',
    stats: 'Hit Rate: ~67% | ROI Promedio: ~8%' 
  },
  v12: { 
    algo: 'Ensemble XGBoost + CatBoost', 
    features: 'Manejo de features categóricas nativas. Alerta: Calculaba "League stats" sobre todo el batch (Data Leakage futuro).',
    samples: '~20,000 partidos en DB',
    dates: 'Hasta Q1 2026 (Filtrado inseguro)',
    timing: 'Live Fijo (minuto 24 para Q3, minuto 36 para Q4).',
    stats: 'Falso Hit Rate ~80%+ (Inoperable real)' 
  },
  v13: { 
    algo: 'Ensemble ponderado por F1 (LogReg, GB, XGB, CatBM)', 
    features: 'Snapshots DINÁMICOS. Partidos segmentados por ritmo (Pace Buckets). Walk-Forward stats genuino sin Leakage.',
    samples: '16,400 partidos validados',
    dates: '22-Ene-2026 a 15-Abr-2026',
    timing: 'Live Dinámico (Cutoffs: min. 18-23 Q3 | min. 28-32 Q4).',
    stats: 'Hit Rate: 52-56% | ROI: 2-5% (Comprobado real)' 
  },
  v15: { 
    algo: 'Micro-Ensembles INDIVIDUALES POR LIGA + Calibración isotónica (CV)', 
    features: '50+ features. Veto cruzado si clasificador vs regresor difiere >4 pts. Gates por liga aprendidos dinámicamente.',
    samples: '~51,000 muestras procesadas (16.6k partidos masc activos)',
    dates: 'Últimos 90 días rolling (17-Ene-2026 a 17-Abr-2026)',
    timing: 'Live Exacto (Justo en minuto 22 para Q3, y minuto 31 para Q4).',
    stats: 'Diseñado a Odds=1.40 | Expected Hit Rate 75%+ | Extrema selectividad' 
  },
  v16: { 
    algo: 'Deep Time Series (TimesFM 200M Google Research) + HistGB/CatBoost', 
    features: 'TimesFM autoregresa diferencias 20 pts adelante en tiempo real. Selector IA de ligas (QA Score). Prunning top 52 feats.',
    samples: '~74k samples ultra limpios / 68+ ligas activas',
    dates: 'Holdout Sweep Vigente: Ene 2026 a 18-Abr-2026',
    timing: 'Live Multi-Snapshot continuo (Cualquier min: >=17 Q3 | >=26 Q4).',
    stats: 'Drift tracking activo | Target Hit Rate ~73%+' 
  },
  v17: { 
    algo: 'Ensemble XGB/CatB/HistGB + Google TimesFM (Deep Time Series Forecasting)', 
    features: 'Audit & Pruning 52 features. Forecasting de serie "score diff" (horizonte 20). Regresión de totales inyectada como feature. Selector inteligente de ligas (Scoring 0-100).',
    samples: '149k samples / 56k post-selección (68 ligas activas)',
    dates: 'Entrenamiento rolling diario (vía .venv311 al 18-Abr-2026)',
    timing: 'Live Multi-Snapshot flexible (Minutos 17, 19, 21, 22, 23 para Q3 | 26 a 33 para Q4).',
    stats: 'Break-even ROI comprobado (+1.82% holdout) | Target Hit Rate 72-76%' 
  }
};

const DailyModelColumn = ({ 
  model, 
  date, 
  autoStart, 
  onComplete,
  rank,
  totalRanked,
  volumeRank,
  onStatsComputed,
  onMinimize,
  totalMatchesDay
}: { 
  model: string; 
  date: string; 
  autoStart: boolean; 
  onComplete: (date: string) => void;
  rank?: number | null;
  totalRanked?: number;
  volumeRank?: number | null;
  onStatsComputed?: (model: string, stats: {roi: number, volume: number} | null) => void;
  onMinimize?: (model: string, isMinimized: boolean) => void;
  totalMatchesDay?: number;
}) => {
  const [data, setData] = useState<MatchData[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(false);
  const [fromCache, setFromCache] = useState(false);
  const [computing, setComputing] = useState(false);
  const [progress, setProgress] = useState<{done:number,total:number,bets:number,pct:number} | null>(null);
  const [hasAutofetchedForDate, setHasAutofetchedForDate] = useState<string | null>(null);
  const [showInfo, setShowInfo] = useState(false);
  const [minimized, setMinimized] = useState(false);
  
  const handleMinimizedChange = useCallback((isMinimized: boolean) => {
    setMinimized(isMinimized);
    onMinimize?.(model, isMinimized);
  }, [model, onMinimize]);

  // Formatea fecha ISO (2026-04-11) a texto legible en español (Sábado 11 Abril 2026)
  const formatDateES = (dateStr: string): string => {
    try {
      const [year, month, day] = dateStr.split('-').map(Number);
      const d = new Date(year, month - 1, day);
      const daysES = ['Domingo', 'Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado'];
      const monthsES = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 
                        'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre'];
      const dayName = daysES[d.getDay()];
      const monthName = monthsES[d.getMonth()];
      return `${dayName} ${day} ${monthName} ${year}`;
    } catch {
      return dateStr;
    }
  };

  const fetchData = useCallback((forceRefresh = false) => {
    setLoading(true);
    setError(false);
    setProgress(null);
    
    const url = `http://localhost:8000/api/compute/${model}/${date}${forceRefresh ? '?force=true' : ''}`;
    
    fetch(url)
      .then(res => {
         const contentType = res.headers.get("content-type");
         if (!res.ok || (contentType && !contentType.includes("application/json"))) {
            throw new Error("Error en API o no es JSON");
         }
         return res.json();
      })
      .then(json => {
        setData(json);
        setFromCache(!forceRefresh);
        setLoading(false);
        setComputing(false);
        setProgress(null);
        if (onComplete) onComplete(date);
      })
      .catch(err => {
        console.error(`Error consultando API para ${model}:`, err);
        setError(true);
        setLoading(false);
        setComputing(false);
        setProgress(null);
        if (onComplete) onComplete(date);
      });
  }, [model, date, onComplete]);

  // Polling del progreso mientras computa
  useEffect(() => {
    if (!loading) return;
    const poll = setInterval(async () => {
      try {
        const res = await fetch(`http://localhost:8000/api/progress/${model}/${date}`);
        const p = await res.json();
        if (p.active) {
          setComputing(true);
          if (p.total > 0) {
            setProgress({ done: p.done, total: p.total, bets: p.bets, pct: p.pct });
          }
        }
      } catch { /* silently ignore */ }
    }, 800);
    return () => clearInterval(poll);
  }, [loading, model, date]);

  useEffect(() => {
    if (autoStart && hasAutofetchedForDate !== date) {
      setHasAutofetchedForDate(date);
      fetchData(false);
    }
  }, [autoStart, date, hasAutofetchedForDate, fetchData]);

  const stats = useMemo(() => {
     if (!data) return null;
     const STAKE = 100, ODDS = 1.4;

     const q3 = data.filter(m => m.target === 'q3');
     const q4 = data.filter(m => m.target === 'q4');

     const calc = (arr: MatchData[]) => {
       let w = 0, l = 0, p = 0, earned = 0;
       arr.forEach(m => {
         if (m.hit && m.outcome !== 'push') { w++; earned += STAKE * (ODDS - 1); }
         else if (m.outcome === 'push') { p++; l++; earned -= STAKE; }
         else { l++; earned -= STAKE; }
       });
       return { w, l, p, total: arr.length, earned,
                pct: arr.length > 0 ? (w / (w + l)) * 100 : 0 };
     };

     const s3 = calc(q3);
     const s4 = calc(q4);
     const sGlobal = {
       w: s3.w + s4.w, l: s3.l + s4.l,
       total: s3.total + s4.total, earned: s3.earned + s4.earned,
       pct: (s3.w + s4.w + s3.l + s4.l) > 0
         ? ((s3.w + s4.w) / (s3.w + s4.w + s3.l + s4.l)) * 100 : 0,
     };

     // Leagues: { league -> { q3w, q3l, q4w, q4l } }
     const leagues: Record<string, {q3w:number,q3l:number,q3p:number,q4w:number,q4l:number,q4p:number}> = {};
     data.forEach(m => {
       if (!leagues[m.league]) leagues[m.league] = {q3w:0,q3l:0,q3p:0,q4w:0,q4l:0,q4p:0};
       const hit = m.outcome === 'push' ? false : m.hit;
       const isPush = m.outcome === 'push';
       if (m.target === 'q3') { 
         if (hit) leagues[m.league].q3w++; 
         else { leagues[m.league].q3l++; if (isPush) leagues[m.league].q3p++; }
       } else { 
         if (hit) leagues[m.league].q4w++; 
         else { leagues[m.league].q4l++; if (isPush) leagues[m.league].q4p++; }
       }
     });
     const sortedLeagues = Object.entries(leagues)
       .sort((a,b) => (b[1].q3l + b[1].q4l) - (a[1].q3l + a[1].q4l));

     return { q3: s3, q4: s4, global: sGlobal, sortedLeagues };
  }, [data]);

  useEffect(() => {
    if (onStatsComputed) {
      if (stats && stats.global.total > 0) {
        // ROI %
        const roi = (stats.global.earned / (stats.global.total * 100)) * 100;
        onStatsComputed(model, { roi, volume: stats.global.total });
      } else {
        onStatsComputed(model, null);
      }
    }
  }, [stats, model, onStatsComputed]);

  // Pre-compute loading spinner values (avoids IIFE inside JSX for OXC compat)
  const _loadingPct = (computing && progress && progress.total > 0) ? progress.pct : null;
  const _loadingR = 52;
  const _loadingC = 2 * Math.PI * _loadingR;
  const _loadingOffset = _loadingPct !== null ? _loadingC * (1 - _loadingPct / 100) : _loadingC;

  if (minimized) {
    return (
      <div className="glass-panel" style={{ width: 48, minWidth: 48, flexShrink: 0, display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '16px 0', cursor: 'pointer', opacity: 0.7 }} onClick={() => handleMinimizedChange(false)} title="Restaurar modelo">
         <div style={{ writingMode: 'vertical-rl', transform: 'rotate(180deg)', fontWeight: 'bold', color: 'var(--text-main)', letterSpacing: 2 }}>{model.toUpperCase()}</div>
         <button className="icon-btn" style={{ marginTop: 'auto' }} onClick={(e) => { e.stopPropagation(); handleMinimizedChange(false); }} title="Mostrar">
           <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path><circle cx="12" cy="12" r="3"></circle></svg>
         </button>
      </div>
    );
  }

  return (
    <div className="glass-panel" style={{ minWidth: 320, maxWidth: 320, flexShrink: 0, display: 'flex', flexDirection: 'column' }}>
       <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '16px', borderBottom: '1px solid var(--border-color)' }}>
         <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
           <h3 style={{ margin: 0, textTransform: 'uppercase', color: 'var(--text-main)' }}>{model}</h3>
           {fromCache && !loading && data && data.length > 0 && (
             <span style={{ fontSize: '0.7rem', background: 'rgba(99,102,241,0.15)', color: 'var(--primary)', padding: '2px 6px', borderRadius: 4, border: '1px solid rgba(99,102,241,0.3)' }}>cache</span>
           )}
         </div>
         <div style={{ display: 'flex', gap: 4 }}>
           <button 
             className="icon-btn" 
             onClick={() => setShowInfo(!showInfo)} 
             title="Ver Info Modelo"
             style={{ color: showInfo ? 'var(--primary)' : 'currentColor' }}
           >
             <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>
           </button>
           <button 
             className="icon-btn" 
             onClick={() => { setComputing(true); fetchData(true); }} 
             title="Recalcular (ignorar cache)"
             disabled={loading}
           >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ color: computing ? 'var(--primary)' : 'currentColor' }}><polyline points="23 4 23 10 17 10"></polyline><polyline points="1 20 1 14 7 14"></polyline><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"></path></svg>
           </button>
           <button 
             className="icon-btn" 
             onClick={() => handleMinimizedChange(true)} 
             title="Ocultar"
             style={{ color: 'var(--text-muted)' }}
           >
             <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24"></path><line x1="1" y1="1" x2="23" y2="23"></line></svg>
           </button>
         </div>
       </div>
       
       <div className="custom-scrollbar" style={{ padding: '16px', flex: 1, overflowY: 'auto' }}>
         {showInfo && (
           <div style={{ background: 'rgba(99,102,241,0.05)', padding: 14, borderRadius: 8, marginBottom: 16, border: '1px solid rgba(99,102,241,0.3)', fontSize: '0.82rem', lineHeight: '1.4' }} className="animate-fade-in">
             <div style={{ fontWeight: 'bold', color: 'var(--primary)', marginBottom: 8, fontSize: '0.9rem', display: 'flex', alignItems: 'center', gap: 6 }}>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2"></path></svg>
                Análisis Deep Dive V{model.replace('v', '')}
             </div>
             <div style={{ color: 'var(--text-main)', marginBottom: 6 }}>
                <strong style={{color: 'rgba(255,255,255,0.7)', display: 'inline-block', width: 90}}>Algoritmos:</strong> 
                {MODEL_INFO[model]?.algo || 'Desconocido'}
             </div>
             <div style={{ color: 'var(--text-main)', marginBottom: 6 }}>
                <strong style={{color: 'rgba(255,255,255,0.7)', display: 'block', marginBottom: 2}}>Features / Enfoque:</strong> 
                {MODEL_INFO[model]?.features || 'N/A'}
             </div>
             <div style={{ color: 'var(--text-main)', marginBottom: 6, display: 'flex', gap: 6, flexDirection: 'column' }}>
                <div><strong style={{color: 'rgba(255,255,255,0.7)', display: 'inline-block', width: 90}}>Registros:</strong> {MODEL_INFO[model]?.samples || 'N/A'}</div>
                <div><strong style={{color: 'rgba(255,255,255,0.7)', display: 'inline-block', width: 90}}>Fechas:</strong> {MODEL_INFO[model]?.dates || 'N/A'}</div>
             </div>
             <div style={{ color: 'var(--primary)', marginBottom: 8, padding: '4px 6px', background: 'rgba(99,102,241,0.1)', borderRadius: 4, display: 'inline-block' }}>
                <strong style={{color: 'rgba(255,255,255,0.8)', marginRight: 4}}>Timing Apuesta:</strong> 
                {MODEL_INFO[model]?.timing || 'N/A'}
             </div>
             <div style={{ color: 'var(--success)' }}>
                <strong style={{color: 'rgba(255,255,255,0.7)', display: 'inline-block', width: 90}}>Benchmark:</strong> 
                {MODEL_INFO[model]?.stats || 'N/A'}
             </div>
           </div>
         )}
         {loading ? (
           <div style={{ display:'flex', flexDirection:'column', alignItems:'center', justifyContent:'center', height:'100%', minHeight:220 }}>
               <div style={{ position:'relative', width:130, height:130 }}>
                 <svg width="130" height="130" viewBox="0 0 130 130" style={{ transform:'rotate(-90deg)' }}>
                   <circle cx="65" cy="65" r={_loadingR} fill="none" stroke="rgba(255,255,255,0.07)" strokeWidth="10"/>
                   {_loadingPct !== null ? (
                     <circle cx="65" cy="65" r={_loadingR} fill="none"
                       stroke="url(#progGrad)" strokeWidth="10" strokeLinecap="round"
                       strokeDasharray={_loadingC} strokeDashoffset={_loadingOffset}
                       style={{ transition:'stroke-dashoffset 0.6s ease' }}
                     />
                   ) : (
                     <circle cx="65" cy="65" r={_loadingR} fill="none"
                       stroke="url(#progGrad)" strokeWidth="10" strokeLinecap="round"
                       strokeDasharray={`${_loadingC * 0.25} ${_loadingC * 0.75}`}
                       style={{ animation:'spin 1s linear infinite', transformOrigin:'65px 65px' }}
                     />
                   )}
                   <defs>
                     <linearGradient id="progGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                       <stop offset="0%" stopColor="var(--primary)"/>
                       <stop offset="100%" stopColor="#818cf8"/>
                     </linearGradient>
                   </defs>
                 </svg>
                 <div style={{ position:'absolute', inset:0, display:'flex', flexDirection:'column', alignItems:'center', justifyContent:'center' }}>
                   <span style={{ fontSize: _loadingPct !== null ? '1.7rem' : '1rem', fontWeight:'bold', color:'var(--text-main)', lineHeight:1 }}>
                     {_loadingPct !== null ? `${_loadingPct}%` : '···'}
                   </span>
                   {_loadingPct !== null && progress && (
                     <span style={{ fontSize:'0.62rem', color:'var(--text-muted)', marginTop:3 }}>
                       {progress.done}/{progress.total}
                     </span>
                   )}
                 </div>
               </div>
               {computing && progress && progress.bets > 0 && (
                 <div style={{ marginTop:14, fontSize:'0.78rem', color:'var(--success)' }}>✓ {progress.bets} BETs</div>
               )}
               <div style={{ marginTop:8, fontSize:'0.75rem', color:'var(--text-muted)' }}>
                 {computing ? `Calculando ${model.toUpperCase()} · ${formatDateES(date)}` : 'Cargando...'}
               </div>
             </div>
         )
         : error || !data ? (
           <div style={{ textAlign: 'center', color: 'var(--text-muted)', marginTop: 40 }}>
             <p>No precomputado</p>
             <button style={{marginTop: 8}} onClick={fetchData} className="tab-btn">Reintentar</button>
           </div>
         ) : !stats || stats.global.total === 0 ? (
           <div style={{ textAlign: 'center', color: 'var(--text-muted)', marginTop: 40 }}>No hay matches para este dia.</div>
         ) : (
           <div className="animate-fade-in">
             {/* ── Tabla Q3 / Q4 / Global ── */}
             <div style={{ background: 'rgba(0,0,0,0.2)', padding: '12px 16px', borderRadius: 8, marginBottom: 16, position: 'relative', overflow: 'hidden' }}>
               {rank != null && rank > 0 && totalRanked != null && (
                 <div style={{ 
                   position: 'absolute', 
                   top: '50%', 
                   right: '-5px', 
                   transform: 'translateY(-50%)', 
                   fontSize: '6.5rem', 
                   fontWeight: 900, 
                   opacity: 0.15, 
                   lineHeight: 1,
                   color: rank === 1 ? '#22c55e' : rank <= Math.ceil(totalRanked / 2) ? '#eab308' : '#ef4444',
                   pointerEvents: 'none',
                   zIndex: 0
                 }}>
                   {rank}
                 </div>
               )}
               {volumeRank != null && volumeRank > 0 && (
                 <div style={{ 
                   position: 'absolute', 
                   top: '50%', 
                   left: '-5px', 
                   transform: 'translateY(-50%)', 
                   fontSize: '6.5rem', 
                   fontWeight: 900, 
                   opacity: 0.08, 
                   lineHeight: 1,
                   color: '#6366f1',
                   pointerEvents: 'none',
                   zIndex: 0
                 }} title="Ranking de Volumen (Total Apuestas)">
                   {volumeRank}
                 </div>
               )}
               <div style={{ position: 'relative', zIndex: 1 }}>
                 <table style={{ width: '100%', fontSize: '0.8rem', borderCollapse: 'collapse' }}>
                   <thead>
                     <tr style={{ color: 'var(--text-muted)', borderBottom: '1px solid rgba(255,255,255,0.08)' }}>
                       <th style={{ textAlign:'left', padding:'4px 4px' }}>Filtro</th>
                       <th style={{ textAlign:'center', padding:'4px 4px', color:'var(--success)' }}>W</th>
                       <th style={{ textAlign:'center', padding:'4px 4px', color:'var(--danger)' }}>L</th>
                       <th style={{ textAlign:'center', padding:'4px 4px' }}>NB</th>
                       <th style={{ textAlign:'right', padding:'4px 4px' }}>%W</th>
                       <th style={{ textAlign:'right', padding:'4px 4px' }}>Gan.</th>
                     </tr>
                   </thead>
                   <tbody>
                     {([['Q3', stats.q3], ['Q4', stats.q4], ['GLOBAL', stats.global]] as const).map(([label, s]) => {
                       const evaluated = label === 'GLOBAL' ? (totalMatchesDay ? totalMatchesDay * 2 : 0) : (totalMatchesDay || 0);
                       const nb = evaluated > 0 ? Math.max(0, evaluated - s.total) : '-';
                       return (
                         <tr key={label} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)', fontWeight: label==='GLOBAL' ? 'bold' : 'normal' }}>
                           <td style={{ padding:'5px 4px', color: label==='GLOBAL' ? 'var(--text-main)' : 'var(--text-muted)' }}>{label}</td>
                           <td style={{ textAlign:'center', color:'var(--success)', padding:'5px 4px' }}>{s.w}</td>
                           <td style={{ textAlign:'center', color:'var(--danger)', padding:'5px 4px' }}>{s.l}</td>
                           <td style={{ textAlign:'center', color:'var(--text-muted)', padding:'5px 4px' }}>{nb}</td>
                           <td style={{ textAlign:'right', padding:'5px 4px' }}>{s.pct.toFixed(1)}%</td>
                           <td style={{ textAlign:'right', padding:'5px 4px', color: s.earned >= 0 ? 'var(--success)' : 'var(--danger)' }}>
                             {s.earned >= 0 ? '+' : ''}${s.earned.toFixed(0)}
                           </td>
                         </tr>
                       );
                     })}
                   </tbody>
                 </table>
               </div>
             </div>

             {/* ── Tabla por liga ── */}
             <h4 style={{ margin: '0 0 10px 0', color: 'var(--text-main)', fontSize: '0.95rem' }}>Por Liga <span style={{fontSize:'0.75rem', color:'var(--text-muted)', fontWeight:'normal'}}>(por pérdidas)</span></h4>
             <table style={{ fontSize: '0.78rem', width: '100%', borderCollapse: 'collapse' }}>
               <thead>
                 <tr style={{ color: 'var(--text-muted)', borderBottom: '1px solid rgba(255,255,255,0.08)' }}>
                   <th style={{ textAlign:'left', padding:'4px 4px' }}>Liga</th>
                   <th style={{ textAlign:'center', padding:'4px 2px', color:'var(--success)' }}>Q3W</th>
                   <th style={{ textAlign:'center', padding:'4px 2px', color:'var(--danger)' }}>Q3L</th>
                   <th style={{ textAlign:'center', padding:'4px 2px', color:'var(--success)' }}>Q4W</th>
                   <th style={{ textAlign:'center', padding:'4px 2px', color:'var(--danger)' }}>Q4L</th>
                   <th style={{ textAlign:'right', padding:'4px 4px' }}>%W</th>
                 </tr>
               </thead>
               <tbody>
                 {stats.sortedLeagues.map(([league, ls], idx) => {
                   const tw = ls.q3w + ls.q4w;
                   const tl = ls.q3l + ls.q4l;
                   const pct = (tw + tl) > 0 ? (tw / (tw + tl)) * 100 : 0;
                   return (
                     <tr key={idx} style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
                       <td style={{ padding:'6px 4px' }}>{league}</td>
                       <td style={{ textAlign:'center', padding:'6px 2px', color:'var(--success)', fontWeight:'bold' }}>{ls.q3w}</td>
                       <td style={{ textAlign:'center', padding:'6px 2px', color:'var(--danger)', fontWeight:'bold' }}>{ls.q3l}</td>
                       <td style={{ textAlign:'center', padding:'6px 2px', color:'var(--success)', fontWeight:'bold' }}>{ls.q4w}</td>
                       <td style={{ textAlign:'center', padding:'6px 2px', color:'var(--danger)', fontWeight:'bold' }}>{ls.q4l}</td>
                       <td style={{ textAlign:'right', padding:'6px 4px', color: pct>=50?'var(--success)':'var(--danger)' }}>{pct.toFixed(0)}%</td>
                     </tr>
                   );
                 })}
               </tbody>
             </table>
           </div>
         )}
       </div>
    </div>
  )
}

function App() {
  const [model, setModel] = useState("v13");
  const [columnsOrder, setColumnsOrder] = useState<string[]>(VERSIONS);
  const [draggedCol, setDraggedCol] = useState<string | null>(null);

  const handleDragStart = (e: React.DragEvent, colModel: string) => {
    setDraggedCol(colModel);
    e.dataTransfer.effectAllowed = 'move';
    e.dataTransfer.setData('text/plain', colModel);
  };
  
  const handleDragOver = (e: React.DragEvent, colModel: string) => {
    e.preventDefault(); 
    if (!draggedCol || draggedCol === colModel) return;
    setColumnsOrder(prev => {
       const newOrder = [...prev];
       const dragIndex = newOrder.indexOf(draggedCol);
       const hoverIndex = newOrder.indexOf(colModel);
       if (dragIndex < 0 || hoverIndex < 0) return prev;
       newOrder.splice(dragIndex, 1);
       newOrder.splice(hoverIndex, 0, draggedCol);
       return newOrder;
    });
  };

  const handleDragEnd = () => {
    setDraggedCol(null);
  };

  const [data, setData] = useState<MatchData[]>([]);
  const [loading, setLoading] = useState(true);

  // Sorting
  const [sortField, setSortField] = useState<keyof MatchData | 'earned'>('date');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');
  
  // Navigation
  const [currentView, setCurrentView] = useState<'matches' | 'leagues' | 'daily'>('daily');
  const [sidebarOpen, setSidebarOpen] = useState<boolean>(() => {
    const saved = localStorage.getItem('sidebarOpen');
    return saved === null ? true : saved === 'true';
  });
  const setSidebarOpenPersist = (val: boolean) => {
    setSidebarOpen(val);
    localStorage.setItem('sidebarOpen', String(val));
  };
  
  // Daily Comparison Date
  const [comparisonDate, setComparisonDate] = useState(() => {
    const today = new Date();
    return `${today.getFullYear()}-${String(today.getMonth() + 1).padStart(2, '0')}-${String(today.getDate()).padStart(2, '0')}`;
  });
  
  const [queueIndex, setQueueIndex] = useState(0);
  const [modelStats, setModelStats] = useState<Record<string, {roi: number, volume: number} | null>>({});
  const [minimizedModels, setMinimizedModels] = useState<Set<string>>(new Set());
  const currentDateRef = React.useRef(comparisonDate);
  const [totalMatchesDay, setTotalMatchesDay] = useState(0);

  const handleMinimize = useCallback((model: string, isMinimized: boolean) => {
    setMinimizedModels(prev => {
      const next = new Set(prev);
      if (isMinimized) {
        next.add(model);
      } else {
        next.delete(model);
      }
      return next;
    });
  }, []);

  useEffect(() => {
    currentDateRef.current = comparisonDate;
    setQueueIndex(0);
    setModelStats({});
    fetch(`http://localhost:8000/api/matches/count/${comparisonDate}`)
      .then(r => r.json())
      .then(d => setTotalMatchesDay(d.count || 0))
      .catch(() => setTotalMatchesDay(0));
  }, [comparisonDate]);

  const handleColumnComplete = useCallback((completedDate: string) => {
    if (currentDateRef.current === completedDate) {
      setQueueIndex(prev => prev + 1);
    }
  }, []);

  const handleStatsComputed = useCallback((m: string, s: {roi: number, volume: number} | null) => {
    setModelStats(prev => {
      if (!s) {
        if (prev[m] === null) return prev;
        return { ...prev, [m]: null };
      }
      if (prev[m] && prev[m]!.roi === s.roi && prev[m]!.volume === s.volume) return prev;
      return { ...prev, [m]: s };
    });
  }, []);

  const rankedModels = useMemo(() => {
    return Object.entries(modelStats)
      .filter(([m, s]) => s != null && !minimizedModels.has(m))
      .sort((a, b) => b[1]!.roi - a[1]!.roi)
      .map(x => x[0]);
  }, [modelStats, minimizedModels]);

  const volumeRankedModels = useMemo(() => {
    return Object.entries(modelStats)
      .filter(([m, s]) => s != null && !minimizedModels.has(m))
      .sort((a, b) => b[1]!.volume - a[1]!.volume)
      .map(x => x[0]);
  }, [modelStats, minimizedModels]);
  
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
        <button className="icon-btn glass-panel" onClick={() => setSidebarOpenPersist(true)} style={{ position: 'absolute', top: 16, left: 16, zIndex: 100, background: 'var(--panel-bg)', padding: 12 }}>
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="3" y1="12" x2="21" y2="12"></line><line x1="3" y1="6" x2="21" y2="6"></line><line x1="3" y1="18" x2="21" y2="18"></line></svg>
        </button>
      )}

      <div className="dashboard-layout" style={{ display: 'flex', overflow: 'hidden', flex: 1 }}>
        {/* Sidebar */}
        <aside className={`glass-panel controls-panel sidebar ${sidebarOpen ? 'open' : 'closed'}`} style={{ zIndex: 20 }}>
          
          {/* Header Controls inside Sidebar */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
             <h1 className="text-gradient" style={{ margin: 0, fontSize: '1.8rem' }}>Pulpa</h1>
             <button className="icon-btn" onClick={() => setSidebarOpenPersist(false)}>
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
              <option value="v17">Modelo V17 (Deep Forecast)</option>
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
            <button 
              className={`sidebar-link ${currentView === 'daily' ? 'active' : ''}`}
              onClick={() => setCurrentView('daily')}
            >
              🗓 Daily Models Comparativa
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
          {/* Metrics - only show on matches and leagues view */}
          {currentView !== 'daily' && (
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
          )}



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

          {/* Daily Comparison View */}
          {currentView === 'daily' && (
            <div className="animate-fade-in" style={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>
              <div style={{ display: 'flex', alignItems: 'center', marginBottom: 24, gap: 16 }}>
                <h2 style={{ margin: 0, color: 'var(--text-main)', fontSize: '1.8rem' }}>ROI Modelos</h2>
                <input 
                  type="date"
                  className="select-input"
                  value={comparisonDate}
                  onChange={e => setComparisonDate(e.target.value)} 
                  style={{ fontSize: '1.1rem', padding: '8px 16px', background: 'rgba(15, 23, 42, 0.95)' }}
                />
                <div style={{ display: 'flex', gap: '8px' }}>
                  <button
                    className="action-btn"
                    onClick={() => {
                      setColumnsOrder(prev => {
                        const rankedSet = new Set(rankedModels);
                        const unranked = prev.filter(m => !rankedSet.has(m));
                        return [...rankedModels, ...unranked];
                      });
                    }}
                    style={{
                      padding: '8px 16px',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '8px',
                      background: 'rgba(34, 197, 94, 0.15)',
                      border: '1px solid rgba(34, 197, 94, 0.3)',
                      color: '#22c55e',
                      borderRadius: '8px',
                      cursor: rankedModels.length > 1 ? 'pointer' : 'not-allowed',
                      opacity: rankedModels.length > 1 ? 1 : 0.5,
                      fontWeight: 'bold',
                      fontSize: '0.85rem'
                    }}
                    title="Ordenar por ROI (Rentabilidad)"
                  >
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon></svg>
                    Ordenar ROI
                  </button>

                  <button
                    className="action-btn"
                    onClick={() => {
                      setColumnsOrder(prev => {
                        const rankedSet = new Set(volumeRankedModels);
                        const unranked = prev.filter(m => !rankedSet.has(m));
                        return [...volumeRankedModels, ...unranked];
                      });
                    }}
                    style={{
                      padding: '8px 16px',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '8px',
                      background: 'rgba(99, 102, 241, 0.15)',
                      border: '1px solid rgba(99, 102, 241, 0.3)',
                      color: '#6366f1',
                      borderRadius: '8px',
                      cursor: volumeRankedModels.length > 1 ? 'pointer' : 'not-allowed',
                      opacity: volumeRankedModels.length > 1 ? 1 : 0.5,
                      fontWeight: 'bold',
                      fontSize: '0.85rem'
                    }}
                    title="Ordenar por Volumen (Cantidad de Apuestas)"
                  >
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path><polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline><line x1="12" y1="22.08" x2="12" y2="12"></line></svg>
                    Ordenar Volumen
                  </button>
                </div>
                <div style={{ marginLeft: 'auto', background: 'rgba(99, 102, 241, 0.1)', padding: '8px 16px', borderRadius: 8, border: '1px solid rgba(99, 102, 241, 0.3)' }}>
                   <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Configuración Fija:</span>
                   <span style={{ marginLeft: 12, fontWeight: 'bold', color: 'var(--text-main)' }}>Bank: $1000 | Apuesta: $100 | Momio: 1.4</span>
                </div>
              </div>
              
              <div 
                className="custom-scrollbar"
                style={{ 
                  display: 'flex', 
                  overflowX: 'auto', 
                  gap: '24px', 
                  flex: 1, 
                  paddingBottom: '16px',
                  alignItems: 'stretch'
                }}
              >
                {columnsOrder.map((v, index) => (
                  <div
                    key={v}
                    draggable
                    onDragStart={(e) => handleDragStart(e, v)}
                    onDragOver={(e) => handleDragOver(e, v)}
                    onDragEnd={handleDragEnd}
                    onDrop={handleDragEnd}
                    style={{ 
                       cursor: draggedCol ? 'grabbing' : 'grab', 
                       display: 'flex', 
                       flexDirection: 'column',
                       opacity: draggedCol === v ? 0.6 : 1,
                       transition: 'opacity 0.2s ease',
                       transform: 'translateZ(0)'
                    }}
                  >
                    <DailyModelColumn 
                      model={v} 
                      date={comparisonDate} 
                      autoStart={index <= queueIndex}
                      onComplete={handleColumnComplete}
                      rank={rankedModels.indexOf(v) >= 0 ? rankedModels.indexOf(v) + 1 : null}
                      volumeRank={volumeRankedModels.indexOf(v) >= 0 ? volumeRankedModels.indexOf(v) + 1 : null}
                      totalRanked={rankedModels.length}
                      onStatsComputed={handleStatsComputed}
                      onMinimize={handleMinimize}
                      totalMatchesDay={totalMatchesDay}
                    />
                  </div>
                ))}
              </div>
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
