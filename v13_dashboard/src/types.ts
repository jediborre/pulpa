export interface MatchData {
  match_id: string;
  target: string;
  date: string;
  league: string;
  volatility: number;
  gender: string;
  pace_bucket: string;
  confidence: number;
  winner_pick: string;
  target_winner: string;
  hit: boolean;
  outcome?: string;  // 'hit' | 'miss' | 'push' | 'pending'
  q1_home: number;
  q1_away: number;
  q2_home: number;
  q2_away: number;
  q3_home: number;
  q3_away: number;
}

export interface GateParams {
  minConfidenceQ3: number;
  minConfidenceQ4: number;
  maxVolatility: number;
  allowWomen: boolean;
  odds: number;
  kellyFraction: number;
  bankroll: number;
  startDate: string;
  endDate: string;
}
