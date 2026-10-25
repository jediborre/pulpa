for f in ['docs/ANALISIS_LIGAS_M27_V5_3.md','docs/ANALISIS_LIGAS_M27_V5_3_1.md','docs/ANALISIS_LIGAS_M27_V5_3_2.md','docs/ANALISIS_LIGAS_M27_V5_3_3.md']:
    s = open(f, encoding='utf-8').read()
    print(f, '-> ?', s.count('| ? |'))
