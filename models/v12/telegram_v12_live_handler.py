async def _handle_v12_live_analysis(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    query,
    match_id: str,
    page: int,
    refresh: bool = False,
):
    """Handle V12 LIVE Bookmaker analysis."""
    try:
        # Get match data
        data = await _get_match_detail_async(match_id, update)
        if not data:
            await query.edit_message_text(
                text=f"Match {match_id} no encontrado.",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("Menu principal", callback_data="nav:main")]
                ]),
            )
            return
        
        # Check if match is in-progress or finished
        status_type = str(data.get("match", {}).get("status_type", "") or "").lower()
        if status_type not in ["inprogress", "finished"]:
            await query.edit_message_text(
                text="V12 LIVE solo funciona para partidos en vivo o terminados.\n"
                     f"Estado actual: {status_type}",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("Volver", callback_data=f"match:{match_id}:_:{page}")]
                ]),
            )
            return
        
        # Analyze both Q3 and Q4
        quarters = data.get("score", {}).get("quarters", {})
        pbp = data.get("play_by_play", {})
        gp = data.get("graph_points", [])
        
        analysis_text = []
        has_any_analysis = False
        
        for quarter_label in ["Q3", "Q4"]:
            q = quarters.get(quarter_label)
            if not q:
                continue
            
            q_home = int(q.get("home", 0))
            q_away = int(q.get("away", 0))
            
            # Calculate cumulative scores
            q_order = ["Q1", "Q2", "Q3", "Q4"]
            q_idx = q_order.index(quarter_label)
            total_home = 0
            total_away = 0
            for i in range(q_idx + 1):
                q_data = quarters.get(q_order[i], {})
                total_home += int(q_data.get("home", 0))
                total_away += int(q_data.get("away", 0))
            
            # Simulate mid-quarter (6 min) for analysis
            elapsed = 6.0
            qtr_home_half = q_home // 2
            qtr_away_half = q_away // 2
            
            # Filter graph points
            cutoff = (q_idx * 12) + elapsed
            gp_filtered = [p for p in gp if int(p.get("minute", 0)) <= cutoff]
            
            # Run analysis
            result = _run_v12_live_analysis(
                match_id=match_id,
                quarter=quarter_label,
                qtr_home_score=qtr_home_half,
                qtr_away_score=qtr_away_half,
                total_home_score=total_home,
                total_away_score=total_away,
                elapsed_minutes=elapsed,
                graph_points=gp_filtered,
                pbp_events=pbp.get(quarter_label, [])[:10],
            )
            
            if result.get("ok"):
                has_any_analysis = True
                analysis_text.append(
                    f"{'='*40}\n"
                    f"{quarter_label} - Marcador: {qtr_home_half}-{qtr_away_half}\n"
                    f"{'='*40}\n"
                    f"Proyección: {result['projections']['home']:.0f}-{result['projections']['away']:.0f} "
                    f"(diff {result['projections']['diff']:+.0f})\n"
                    f"Momentum: {result['momentum']:+.1f}\n"
                    f"Total proyectado: {result['projections']['total']:.0f} pts\n\n"
                    f"{result['markets_text']}"
                )
        
        if not has_any_analysis:
            await query.edit_message_text(
                text="No hay datos suficientes para analizar.",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("Volver", callback_data=f"match:{match_id}:_:{page}")]
                ]),
            )
            return
        
        # Build message
        message = (
            f"V12 LIVE BOOKMAKER - Match {match_id}\n\n"
            f"{chr(10).join(analysis_text)}\n\n"
            f"{'='*40}\n"
            f"NOTA: Compara estas líneas justas con tu casa de apuestas.\n"
            f"Si tu casa ofrece MEJORES odds que las 'fair_odds' + 15%,\n"
            f"hay VALUE. Si ofrece PEORES, NO apostes."
        )
        
        # Build keyboard
        keyboard = [
            [InlineKeyboardButton(
                "Refresh análisis",
                callback_data=f"v12live:refresh:{match_id}:_:{page}",
            )],
            [InlineKeyboardButton(
                "Volver al match",
                callback_data=f"match:{match_id}:_:{page}",
            )],
            [InlineKeyboardButton("Menu principal", callback_data="nav:main")],
        ]
        
        await query.edit_message_text(
            text=message,
            reply_markup=InlineKeyboardMarkup(keyboard),
        )
        
    except Exception as exc:
        await query.edit_message_text(
            text=f"Error en V12 LIVE: {exc}",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("Volver", callback_data=f"match:{match_id}:_:{page}")]
            ]),
        )
