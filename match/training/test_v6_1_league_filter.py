#!/usr/bin/env python3
"""
test_v6_1_league_filter.py - Test suite para el filtro de ligas de v6_1.

Prueba:
1. Carga del filtro y datos de soporte
2. Criterios de filtrado (hard filters)
3. Warnings (soft filters)
4. Whitelist/Blacklist
5. Configuración desde JSON
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.v6_1_league_filter import LeagueFilter, LeagueFilterConfig


def test_filter_loads():
    """Test: Filtro carga sin errores."""
    print("\n" + "="*70)
    print("TEST 1: League Filter Loads")
    print("="*70)
    
    try:
        league_filter = LeagueFilter.load()
        print(f"✅ Filter loaded successfully")
        print(f"   - Q3 leagues loaded: {len(league_filter.metrics_by_target['q3'])}")
        print(f"   - Q4 leagues loaded: {len(league_filter.metrics_by_target['q4'])}")
        print(f"   - Filter enabled: {league_filter.config.enabled}")
        return True
    except Exception as e:
        print(f"❌ Failed to load filter: {e}")
        return False


def test_whitelisted_league():
    """Test: Liga blanca nunca se filtra."""
    print("\n" + "="*70)
    print("TEST 2: Whitelisted League Never Filtered")
    print("="*70)
    
    league_filter = LeagueFilter.load()
    
    for league in league_filter.config.whitelist_leagues:
        should_filter, reason = league_filter.should_filter(
            league_bucket=league,
            target="q4",
            model_confidence=0.5,
        )
        status = "✅" if not should_filter else "❌"
        print(f"{status} {league}: filtered={should_filter}")
        if should_filter:
            return False
    
    return True


def test_blacklisted_league():
    """Test: Liga negra siempre se filtra."""
    print("\n" + "="*70)
    print("TEST 3: Blacklisted League Always Filtered")
    print("="*70)
    
    config = LeagueFilterConfig(
        enabled=True,
        blacklist_leagues={"TestLeague"}
    )
    league_filter = LeagueFilter(config)
    league_filter._loaded = True  # Pretend loaded
    
    should_filter, reason = league_filter.should_filter(
        league_bucket="TestLeague",
        target="q4",
        model_confidence=0.5,
    )
    
    if should_filter:
        print(f"✅ Blacklisted league filtered: {reason}")
        return True
    else:
        print(f"❌ Blacklisted league NOT filtered")
        return False


def test_league_not_in_training():
    """Test: Liga no entrenada se filtra."""
    print("\n" + "="*70)
    print("TEST 4: Unknown League Filtered")
    print("="*70)
    
    league_filter = LeagueFilter.load()
    
    should_filter, reason = league_filter.should_filter(
        league_bucket="FAKE_LEAGUE_XYZ_UNKNOWN",
        target="q4",
        model_confidence=0.5,
    )
    
    if should_filter:
        print(f"✅ Unknown league filtered: {reason}")
        return True
    else:
        print(f"❌ Unknown league NOT filtered")
        return False


def test_league_with_no_test():
    """Test: Liga sin test set se filtra."""
    print("\n" + "="*70)
    print("TEST 5: League with No Test Set Filtered")
    print("="*70)
    
    league_filter = LeagueFilter.load()
    
    # Buscar una liga con test_rows=0
    no_test_leagues = []
    for league_bucket, metrics in league_filter.metrics_by_target["q4"].items():
        if metrics.test_rows == 0:
            no_test_leagues.append(league_bucket)
    
    if not no_test_leagues:
        print("⚠️  No leagues with zero test set found in Q4")
        return None  # Skip test
    
    league = no_test_leagues[0]
    should_filter, reason = league_filter.should_filter(
        league_bucket=league,
        target="q4",
        model_confidence=0.5,
    )
    
    if should_filter:
        print(f"✅ League with no test set filtered: {league}")
        print(f"   Reason: {reason}")
        return True
    else:
        print(f"❌ League with no test set NOT filtered: {league}")
        return False


def test_league_with_few_rows():
    """Test: Liga con muy pocos datos se filtra."""
    print("\n" + "="*70)
    print("TEST 6: League with Few Rows Filtered")
    print("="*70)
    
    # Usar configuración restrictiva
    config = LeagueFilterConfig(min_total_rows=500)
    league_filter = LeagueFilter.load(config)
    
    # Buscar una liga con pocos datos
    for league_bucket, metrics in league_filter.metrics_by_target["q4"].items():
        if metrics.total_rows < 500:
            should_filter, reason = league_filter.should_filter(
                league_bucket=league_bucket,
                target="q4",
                model_confidence=0.5,
            )
            
            if should_filter:
                print(f"✅ Small league filtered: {league_bucket}")
                print(f"   total_rows={metrics.total_rows}, min_required=500")
                print(f"   Reason: {reason}")
                return True
            else:
                print(f"❌ Small league NOT filtered: {league_bucket}")
                return False
    
    print("⚠️  All leagues have >= 500 rows (unusual)")
    return None


def test_nba_accepted():
    """Test: NBA tiene muchos datos y se acepta."""
    print("\n" + "="*70)
    print("TEST 7: NBA Accepted (Sufficient Data)")
    print("="*70)
    
    league_filter = LeagueFilter.load()
    
    should_filter, reason = league_filter.should_filter(
        league_bucket="NBA",
        target="q4",
        model_confidence=0.6,
    )
    
    if not should_filter:
        metrics = league_filter.metrics_by_target["q4"].get("NBA")
        if metrics:
            print(f"✅ NBA accepted")
            print(f"   total_rows={metrics.total_rows}")
            print(f"   test_rows={metrics.test_rows}")
            print(f"   target_positive_rate={metrics.target_positive_rate:.2%}")
        return True
    else:
        print(f"❌ NBA filtered: {reason}")
        return False


def test_config_from_json():
    """Test: Configuración se carga desde JSON."""
    print("\n" + "="*70)
    print("TEST 8: Configuration Loaded from JSON")
    print("="*70)
    
    try:
        league_filter = LeagueFilter.load()
        print(f"✅ Config loaded successfully")
        print(f"   - enabled: {league_filter.config.enabled}")
        print(f"   - min_total_rows: {league_filter.config.min_total_rows}")
        print(f"   - min_test_rows_hard: {league_filter.config.min_test_rows_hard}")
        print(f"   - min_test_ratio: {league_filter.config.min_test_ratio}")
        print(f"   - whitelisted leagues: {len(league_filter.config.whitelist_leagues)}")
        return True
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return False


def test_can_disable_filter():
    """Test: Filtro se puede desactivar."""
    print("\n" + "="*70)
    print("TEST 9: Filter Can Be Disabled")
    print("="*70)
    
    config = LeagueFilterConfig(enabled=False)
    league_filter = LeagueFilter.load(config)
    
    # Intentar filtrar una liga (debe devolver False)
    should_filter, reason = league_filter.should_filter(
        league_bucket="UNKNOWN_LEAGUE",
        target="q4",
        model_confidence=0.5,
    )
    
    if not should_filter:
        print(f"✅ Filter disabled successfully")
        print(f"   Unknown league accepted when filter is disabled")
        return True
    else:
        print(f"❌ Filter still active when disabled")
        return False


def main():
    """Ejecutar todos los tests."""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "V6_1 LEAGUE FILTER TEST SUITE" + " "*24 + "║")
    print("╚" + "="*68 + "╝")
    
    tests = [
        ("Loads", test_filter_loads),
        ("Whitelist", test_whitelisted_league),
        ("Blacklist", test_blacklisted_league),
        ("Unknown League", test_league_not_in_training),
        ("No Test Set", test_league_with_no_test),
        ("Few Rows", test_league_with_few_rows),
        ("NBA Accepted", test_nba_accepted),
        ("JSON Config", test_config_from_json),
        ("Can Disable", test_can_disable_filter),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ Exception in {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Resumen
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, r in results if r is True)
    failed = sum(1 for _, r in results if r is False)
    skipped = sum(1 for _, r in results if r is None)
    
    for name, result in results:
        if result is True:
            status = "✅ PASS"
        elif result is False:
            status = "❌ FAIL"
        else:
            status = "⊘ SKIP"
        print(f"{status:10} {name}")
    
    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")
    
    if failed == 0:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
