import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates


def lineplot_years_tradevalues1(spalte_of_interest='Commodity Code', what=1234): #einzeln Export

    # df = pd.read_excel('Daten.xlsx', sheet_name='Daten', engine='openpyxl', nrows=1000)
    # df = pd.read_csv('Daten-Tabelle_2.csv', delimiter=';')

    df = pd.read_csv('AUT-Sugar.csv', delimiter=';')
    uq_idx = [f"{df.loc[i, 'Partner ISO']}_{df.loc[i, 'Year']}" for i in df.index]
    df.index = uq_idx

    print(np.unique(df['Commodity']))
    print(np.unique(df['Year']))
    print(np.unique(df['Partner']))
    print(np.unique(df['Trade Flow']))

    # ab 2001
    df = df[df['Year'] >= 2001]
    # ohne world
    df = df[df['Partner ISO'] != 'WLD']

    # ohne DEU und CZE für Bier
    #df = df[df['Partner ISO'] != 'DEU']
    #df = df[df['Partner ISO'] != 'CZE']

    # export und import des jeweiligen commodity codes
    # df_world_e = df[(df[spalte_of_interest] == what) & (df['Trade Flow'] == 'Export')]
    mode = 'Export'
    df_world_i = df[(df[spalte_of_interest] == what) & (df['Trade Flow'] == mode)]

    # df_world_ex = df_world_e['Trade Value (US$)'] / 10**6
    df_world_i['Trade Value (Mio. US$)'] = df_world_i['Trade Value (US$)'] / 10**6

    # absteigend sortieren nach trade value
    # df_world_i = df_world_i.sort_values(by='Trade Value (US$)', ascending=False)

    # laender alphabetisch sortieren damit farben gleich sind
    df_world_i = df_world_i.sort_values(by='Partner ISO')
    pal = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']    # aut not in this plot --> start with C1

    chart = sns.lineplot(x="Year", y="Trade Value (Mio. US$)",
                 hue="Partner ISO", style="Partner ISO",
                 markers=True, dashes=False,
                 data=df_world_i, palette=pal, legend=False)

    what_name = np.unique(df_world_i.loc[df_world_i[spalte_of_interest]==what, 'Commodity'])[0]
    plt.title(f"{mode}, Commodity: \n {what_name}")
    plt.xlabel("Jahr")
    plt.ylabel("Handelswert in Mio. US$")
    fig = plt.gcf()
    fig.set_size_inches(8, 4)
    plt.xticks(rotation=45)
    plt.grid()

    # xaxis interval (requires mdates locator because it's a date)
    locator = mdates.DayLocator(interval=1)
    chart.xaxis.set_major_locator(locator)
    # yaxis interval (not a date but regular numbers)
    ax = plt.gca()
    #ax.yaxis.set_ticks(np.arange(0, df_world['Trade Value (US$)'].max(), 100))

    fig.tight_layout()
    plt.savefig(f'./graphics/col_Exp_value_{what}.png', dpi=300)
    plt.show()


def lineplot_years_tradevalues2(spalte_of_interest='Commodity Code', what=1123): #differenz

    # df = pd.read_excel('Daten.xlsx', sheet_name='Daten', engine='openpyxl', nrows=1000)
    # df = pd.read_csv('Daten-Tabelle_2.csv', delimiter=';')

    df = pd.read_csv('AUT-Sugar.csv', delimiter=';')
    uq_idx = [f"{df.loc[i, 'Partner ISO']}_{df.loc[i, 'Year']}" for i in df.index]
    df.index = uq_idx

    print(np.unique(df['Commodity']))
    print(np.unique(df['Year']))
    print(np.unique(df['Partner']))
    print(np.unique(df['Trade Flow']))

    # ab 2001
    df = df[df['Year'] >= 2001]
    # ohne world
    df = df[df['Partner ISO'] != 'WLD']

    # export und import des jeweiligen commodity codes
    df_world_e = df[(df[spalte_of_interest] == what) & (df['Trade Flow'] == 'Export')]
    df_world_i = df[(df[spalte_of_interest] == what) & (df['Trade Flow'] == 'Import')]
    print(df_world_e.shape, df_world_i.shape)   # muss gleich sein
    gemeinsamer_index = np.intersect1d(df_world_e.index, df_world_i.index)
    df_e = df_world_e.loc[gemeinsamer_index]
    df_i = df_world_i.loc[gemeinsamer_index]

    df_world = df_e.copy()
    df_world['Export-Import (US$)'] = df_e['Trade Value (US$)'].values - df_i['Trade Value (US$)'].values
    df_world['Export-Import (Mio. US$)'] = df_world['Export-Import (US$)'] / 10**6
    # absteigend sortieren nach trade value
    # df_world = df_world.sort_values(by='Export-Import (Mio. US$)', ascending=False)

    # laender alphabetisch sortieren damit farben gleich sind
    df_world_i = df_world_i.sort_values(by='Partner ISO')
    pal = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']    # aut not in this plot --> start with C1

    chart = sns.lineplot(x="Year", y='Export-Import (Mio. US$)',
                 hue="Partner ISO", style="Partner ISO",
                 markers=True, dashes=False,
                 data=df_world, palette=pal, legend=False)

    what_name = np.unique(df_world.loc[df_world[spalte_of_interest] == what, 'Commodity'])[0]
    plt.title(f"Export-Import von/nach AUT, \n {spalte_of_interest}: {what_name}")
    plt.xlabel("Jahr")
    plt.ylabel("Export - Import (Mio. US$)")
    fig = plt.gcf()
    fig.set_size_inches(8, 4)
    plt.xticks(rotation=45)
    plt.grid()

    # xaxis interval (requires mdates locator because it's a date)
    locator = mdates.DayLocator(interval=1)
    chart.xaxis.set_major_locator(locator)

    # yaxis interval (not a date but regular numbers)
    ax = plt.gca()
    #ax.yaxis.set_ticks(np.arange(0, df_world['Trade Value (US$)'].max(), 100))

    fig.tight_layout()
    plt.savefig(f'./graphics/col_Exp_minus_Imp_{what}.png', dpi=300)
    plt.show()


def lineplot_years_tradevalues3(spalte_of_interest='Commodity Code', what=1123): #differenz zwischen unterschl Ökonomien

    # df = pd.read_excel('Daten.xlsx', sheet_name='Daten', engine='openpyxl', nrows=1000)
    # df = pd.read_csv('Daten-Tabelle_2.csv', delimiter=';')

    df = pd.read_csv('AUT-Sugar.csv', delimiter=';')
    uq_idx = [f"{df.loc[i, 'Partner ISO']}_{df.loc[i, 'Year']}" for i in df.index]
    df.index = uq_idx

    # ab 2001
    df = df[df['Year'] >= 2001]
    # ohne world
    df = df[df['Partner ISO'] != 'WLD']

    # export und import des jeweiligen commodity codes
    df_world_e = df[(df[spalte_of_interest] == what) & (df['Trade Flow'] == 'Export')]
    df_world_i = df[(df[spalte_of_interest] == what) & (df['Trade Flow'] == 'Import')]
    print(df_world_e.shape, df_world_i.shape)   # muss gleich sein
    gemeinsamer_index = np.intersect1d(df_world_e.index, df_world_i.index)
    df_e = df_world_e.loc[gemeinsamer_index]
    df_i = df_world_i.loc[gemeinsamer_index]

    df_world = df_e.copy()
    df_world['Export-Import norm.'] = (df_e['Trade Value (US$)'].values - df_i['Trade Value (US$)'].values)/(df_e['Trade Value (US$)'].values + df_i['Trade Value (US$)'].values)
    # df_world['Export-Import (Mio. US$) norm.'] = df_world['Export-Import (US$) norm.'] / 10**6
    # absteigend sortieren nach trade value
    # df_world = df_world.sort_values(by='Export-Import norm.', ascending=False)

    # laender alphabetisch sortieren damit farben gleich sind
    df_world = df_world.sort_values(by='Partner ISO')
    pal = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']    # aut not in this plot --> start with C1

    chart = sns.lineplot(x="Year", y="Export-Import norm.",
                 hue="Partner ISO", style="Partner ISO",
                 markers=True, dashes=False,
                 data=df_world, palette=pal, legend=False)

    what_name = np.unique(df_world.loc[df_world[spalte_of_interest] == what, 'Commodity'])[0]
    plt.title(f"Nettoaussenhandelsposition (Exp-Imp nach/von AUT), \n {spalte_of_interest}: {what_name}")
    plt.xlabel("Jahr")
    plt.ylabel("Export - Import norm.")
    fig = plt.gcf()
    fig.set_size_inches(8, 4)
    plt.xticks(rotation=45)
    plt.grid()

    # xaxis interval (requires mdates locator because it's a date)
    locator = mdates.DayLocator(interval=1)
    chart.xaxis.set_major_locator(locator)

    # yaxis interval (not a date but regular numbers)
    ax = plt.gca()
    #ax.yaxis.set_ticks(np.arange(0, df_world['Trade Value (US$)'].max(), 100))

    fig.tight_layout()
    plt.savefig(f'./graphics/col_Exp_minus_Imp_norm_{what}.png', dpi=300)
    plt.show()


def lineplot_years_tradevalues4(spalte_of_interest='Commodity Code', what=1123): #Export + Import

    # df = pd.read_excel('Daten.xlsx', sheet_name='Daten', engine='openpyxl', nrows=1000)
    # df = pd.read_csv('Daten-Tabelle_2.csv', delimiter=';')

    df = pd.read_csv('AUT-Sugar.csv', delimiter=';')
    uq_idx = [f"{df.loc[i, 'Partner ISO']}_{df.loc[i, 'Year']}" for i in df.index]
    df.index = uq_idx

    print(np.unique(df['Commodity']))
    print(np.unique(df['Year']))
    print(np.unique(df['Partner']))
    print(np.unique(df['Trade Flow']))

    # ab 2001
    df = df[df['Year'] >= 2001]
    # ohne world
    df = df[df['Partner ISO'] != 'WLD']

    # export und import des jeweiligen commodity codes
    df_world_e = df[(df[spalte_of_interest] == what) & (df['Trade Flow'] == 'Export')]
    df_world_i = df[(df[spalte_of_interest] == what) & (df['Trade Flow'] == 'Import')]
    print(df_world_e.shape, df_world_i.shape)   # muss gleich sein
    gemeinsamer_index = np.intersect1d(df_world_e.index, df_world_i.index)
    df_e = df_world_e.loc[gemeinsamer_index]
    df_i = df_world_i.loc[gemeinsamer_index]

    df_world = df_e.copy()
    df_world['Export-Import (US$)'] = df_e['Trade Value (US$)'].values + df_i['Trade Value (US$)'].values
    df_world['Export-Import (Mio. US$)'] = df_world['Export-Import (US$)'] / 10**6
    # absteigend sortieren nach trade value
    # df_world = df_world.sort_values(by='Export-Import (Mio. US$)', ascending=False)

     # laender alphabetisch sortieren damit farben gleich sind
    df_world = df_world.sort_values(by='Partner ISO')
    pal = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']    # aut not in this plot --> start with C1

    chart = sns.lineplot(x="Year", y='Export-Import (Mio. US$)',
                 hue="Partner ISO", style="Partner ISO",
                 markers=True, dashes=False,
                 data=df_world, palette=pal, legend=False)

    what_name = np.unique(df_world.loc[df_world[spalte_of_interest] == what, 'Commodity'])[0]
    plt.title(f"Export+Import von/nach AUT, \n {spalte_of_interest}: {what_name}")
    plt.xlabel("Jahr")
    plt.ylabel("Export - Import (Mio. US$)")
    fig = plt.gcf()
    fig.set_size_inches(8, 4)
    plt.xticks(rotation=45)
    plt.grid()

    # xaxis interval (requires mdates locator because it's a date)
    locator = mdates.DayLocator(interval=1)
    chart.xaxis.set_major_locator(locator)

    # yaxis interval (not a date but regular numbers)
    ax = plt.gca()
    #ax.yaxis.set_ticks(np.arange(0, df_world['Trade Value (US$)'].max(), 100))

    fig.tight_layout()
    plt.savefig(f'./graphics/col_Exp_plus_Imp_{what}.png', dpi=300)
    plt.show()


def lineplot_years_tradevalues5(spalte_of_interest='Commodity Code', what=1234): #einzeln Import

    # df = pd.read_excel('Daten.xlsx', sheet_name='Daten', engine='openpyxl', nrows=1000)
    # df = pd.read_csv('Daten-Tabelle_2.csv', delimiter=';')

    df = pd.read_csv('AUT-Sugar.csv', delimiter=';')
    uq_idx = [f"{df.loc[i, 'Partner ISO']}_{df.loc[i, 'Year']}" for i in df.index]
    df.index = uq_idx

    print(np.unique(df['Commodity']))
    print(np.unique(df['Year']))
    print(np.unique(df['Partner']))
    print(np.unique(df['Trade Flow']))

    # ab 2001
    df = df[df['Year'] >= 2001]
    # ohne world
    df = df[df['Partner ISO'] != 'WLD']

    # ohne DEU und CZE für Bier
    #df = df[df['Partner ISO'] != 'DEU']
    #df = df[df['Partner ISO'] != 'CZE']

    # export und import des jeweiligen commodity codes
    # df_world_e = df[(df[spalte_of_interest] == what) & (df['Trade Flow'] == 'Export')]
    mode = 'Import'
    df_world_i = df[(df[spalte_of_interest] == what) & (df['Trade Flow'] == mode)]

    # df_world_ex = df_world_e['Trade Value (US$)'] / 10**6
    df_world_i['Trade Value (Mio. US$)'] = df_world_i['Trade Value (US$)'] / 10**6

    # absteigend sortieren nach trade value
    # df_world_i = df_world_i.sort_values(by='Trade Value (US$)', ascending=False)

    # laender alphabetisch sortieren damit farben gleich sind
    df_world_i = df_world_i.sort_values(by='Partner ISO')
    pal = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']    # aut not in this plot --> start with C1

    chart = sns.lineplot(x="Year", y="Trade Value (Mio. US$)",
                 hue="Partner ISO", style="Partner ISO",
                 markers=True, dashes=False,
                 data=df_world_i, palette=pal, legend=False)

    what_name = np.unique(df_world_i.loc[df_world_i[spalte_of_interest]==what, 'Commodity'])[0]
    plt.title(f"{mode}, Commodity: \n {what_name}")
    plt.xlabel("Jahr")
    plt.ylabel("Handelswert in Mio. US$")
    fig = plt.gcf()
    fig.set_size_inches(8, 4)
    plt.xticks(rotation=45)
    plt.grid()

    # xaxis interval (requires mdates locator because it's a date)
    locator = mdates.DayLocator(interval=1)
    chart.xaxis.set_major_locator(locator)
    # yaxis interval (not a date but regular numbers)
    ax = plt.gca()
    #ax.yaxis.set_ticks(np.arange(0, df_world['Trade Value (US$)'].max(), 100))

    fig.tight_layout()
    plt.savefig(f'./graphics/col_Imp_value_{what}.png', dpi=300)
    plt.show()


def lineplot_alle_zu_welt(spalte_of_interest='Commodity Code', what=1123, in_percent=False):
    df = pd.read_csv('WORLD-Sugar.csv', delimiter=';')

    # ab 2001
    df = df[df['Year'] >= 2001]
    # nur partnerlaender
    countries = ['ITA', 'DEU', 'SVN', 'HUN', 'SVK', 'CHE', 'CZE', 'CHN', 'USA', 'AUT']
    use_rows = [i for i in df.index if df.loc[i, 'Reporter ISO'] in countries]
    df = df.loc[use_rows]

    # nur export, nur commodity wie angegeben
    mode = 'Export'
    df_world_i = df[(df[spalte_of_interest] == what) & (df['Trade Flow'] == mode)]
    # umrechnen in millionen us$
    df_world_i['Trade Value (Mio. US$)'] = df_world_i['Trade Value (US$)'] / 10**6

    # absteigend sortieren nach trade value
    # df_world_i = df_world_i.sort_values(by='Trade Value (US$)', ascending=False)
    # laender alphabetisch sortieren damit farben gleich sind
    df_world_i = df_world_i.sort_values(by='Reporter ISO')
    plot_variable = 'Trade Value (Mio. US$)'

    # zusaetzlich gesamtexport ausrechnen
    # jaehrlicher gesamtexport an WORLD (aber nur von den oben ausgewaehlten Partnern)
    yearly_export = df_world_i[['Year', 'Trade Value (Mio. US$)']].groupby('Year').sum()

    pal = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']    # aut also included in plot

    # plot erstellen
    chart = sns.lineplot(x="Year", y=plot_variable,
                 hue="Reporter ISO", style="Reporter ISO",
                 markers=True, dashes=False,
                 data=df_world_i, palette=pal)
    chart.plot(yearly_export, color='k', linewidth=1.2, label='Gesamt')
    plt.legend()

    what_name = np.unique(df_world_i.loc[df_world_i[spalte_of_interest]==what, 'Commodity'])[0]
    plt.title(f"{mode} an WORLD, Commodity: \n {what_name}")
    plt.xlabel("Jahr")
    plt.ylabel(plot_variable)
    fig = plt.gcf()
    fig.set_size_inches(8, 4)
    plt.xticks(rotation=45)
    plt.grid()

    # xaxis interval (requires mdates locator because it's a date)
    locator = mdates.DayLocator(interval=1)
    chart.xaxis.set_major_locator(locator)
    # yaxis interval (not a date but regular numbers)
    ax = plt.gca()
    #ax.yaxis.set_ticks(np.arange(0, df_world['Trade Value (US$)'].max(), 100))

    fig.tight_layout()
    plt.savefig(f'./graphics/col_Exp_an_WORLD_{what}.png', dpi=300)
    plt.show()


def barchart_alle_zu_welt(spalte_of_interest='Commodity Code', what=1123, in_percent=True):
    df = pd.read_csv('WORLD-Sugar.csv', delimiter=';')

    # ab 2001
    df = df[df['Year'] >= 2001]
    # nur partnerlaender
    countries = ['ITA', 'DEU', 'SVN', 'HUN', 'SVK', 'CHE', 'CZE', 'CHN', 'USA', 'AUT']
    use_rows = [i for i in df.index if df.loc[i, 'Reporter ISO'] in countries]
    df = df.loc[use_rows]

    # nur export, nur commodity wie angegeben
    mode = 'Export'
    df_world_i = df[(df[spalte_of_interest] == what) & (df['Trade Flow'] == mode)]
    # umrechnen in millionen us$
    df_world_i['Trade Value (Mio. US$)'] = df_world_i['Trade Value (US$)'] / 10**6

    # absteigend sortieren nach trade value
    df_world_i = df_world_i.sort_values(by='Trade Value (US$)', ascending=False)
    plot_variable = 'Trade Value (Mio. US$)'

    if in_percent is True:
        plot_variable = 'Gesamtexport'
        # jaehrlicher gesamtexport an WORLD (aber nur von den oben ausgewaehlten Partnern)
        yearly_export = df_world_i[['Year', 'Trade Value (US$)']].groupby('Year').sum()
        # jaehrlichen gesamtexport hinzufuegen zu originalem df & jeweiligem Land
        df_world_i['Gesamtexport'] = -99    # dummy value
        for year in np.unique(df_world_i['Year']):
            idx = np.where(df_world_i['Year'] == year)[0]
            df_world_i['Gesamtexport'].iloc[idx] = df_world_i['Trade Value (US$)'].iloc[idx].values / yearly_export.loc[year].values[0] * 100

    # sortieren, damit groesster exporteur ganz unten
    df_world_i.sort_values(by='Gesamtexport', inplace=True)
    indexes = np.unique(df_world_i['Reporter ISO'], return_index=True)[1]
    reps = [df_world_i['Reporter ISO'].values[index] for index in sorted(indexes)][::-1]

    # tabelle fuer report
    years = [2001, 2005, 2010, 2015, 2020]
    ctrs = ['AUT', 'CZE', 'CHN', 'USA', 'DEU']
    subset = [i for i in df_world_i.index if df_world_i.loc[i, 'Year'] in years and df_world_i.loc[i, 'Reporter ISO'] in ctrs]
    sub_df = df_world_i.loc[subset]
    sub_pivot = pd.pivot(sub_df, index='Year', columns='Reporter ISO', values='Gesamtexport')
    print(sub_pivot.round(3))

    # plot erstellen
    country_color_order = ['AUT', 'CHE', 'CHN', 'CZE', 'DEU', 'HUN', 'ITA', 'SVK', 'SVN', 'USA']
    colors = {}
    for i, ctr in enumerate(country_color_order):
        colors[ctr] = f'C{i}'
    add = np.zeros(len(np.unique(df_world_i['Year'])))
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    for reporter in reps:
        df_rep = df_world_i[df_world_i['Reporter ISO'] == reporter].sort_values(by='Year')
        ax.bar(df_rep['Year'], df_rep[plot_variable], width=0.35, bottom=add, label=reporter, color=colors[reporter])
        add = np.add(add, df_rep[plot_variable].values)

    ax.xaxis.set_ticks(np.arange(2001, 2021, 1))
    plt.legend(loc=1, bbox_to_anchor=(0.76, 0, 0.4, 1))
    what_name = np.unique(df_world_i.loc[df_world_i[spalte_of_interest] == what, 'Commodity'])[0]
    plt.title(f"{mode} an WORLD, Commodity: \n {what_name}")
    plt.xlabel("Jahr")
    plt.ylabel(f'{plot_variable} (%)')
    plt.xticks(rotation=45)
    ax.grid()
    # gitterlinien hinter bars
    ax.set_axisbelow(True)

    plt.subplots_adjust(left=0.1, bottom=0.16, right=0.86, top=0.88)
    plt.savefig(f'./graphics/Exp_an_WORLD_rel_{what}.png', dpi=300)
    plt.show()


def weltanteil(spalte_of_interest='Commodity Code', what=1123, ctr='AUT'):
    # 1123, 11101, 11102 - export aller laender zu WORLD
    df_unter = pd.read_csv('WORLD-Sugar.csv', delimiter=';')
    # summe von 111, 112 (non-alc, alc) aller laender zu WORLd
    df_ober = pd.read_csv('WORLD-Sugar-ALC+NonALC.csv', delimiter=';')

    # ab 2001
    dfu = df_unter[df_unter['Year'] >= 2001]
    dfo = df_ober[df_ober['Year'] >= 2001]

    # nur partnerlaender
    countries = ['ITA', 'DEU', 'SVN', 'HUN', 'SVK', 'CHE', 'CZE', 'CHN', 'USA', 'AUT']
    for df in [dfu, dfo]:
        use_rows = [i for i in df.index if df.loc[i, 'Reporter ISO'] in countries]
        df = df.loc[use_rows]

    # dfu hat import und export
    # gleichzeitig reduktion auf commodity of interest
    index = np.where((dfu[spalte_of_interest] == what) & (dfu['Trade Flow'] == 'Export'))[0]
    dfu = dfu.iloc[index]

    # alle laender - commodity (3. teil vom bruch)
    world_comm = dfu[['Year', 'Trade Value (US$)']].groupby('Year').sum()

    # nur oesterreich - commodity (1. teil vom bruch)
    oe_comm = dfu.loc[dfu['Reporter ISO'] == ctr, ['Year', 'Trade Value (US$)']]
    oe_comm.index = oe_comm['Year']
    oe_comm = oe_comm['Trade Value (US$)'].sort_index()

    # oe - export aller getraenke (2. teil vom bruch)
    oe_group = dfo.loc[dfo['Reporter ISO'] == ctr, ['Year', 'Trade Value (US$)']].groupby('Year').sum()

    # world - export aller getraenke (4. teil vom bruch)
    world_group = dfo[['Year', 'Trade Value (US$)']].groupby('Year').sum()

    bruch_oben = np.divide(oe_comm.values, oe_group.values.flatten())
    bruch_unten = np.divide(world_comm.values, world_group.values).flatten()
    rel_wma = np.divide(bruch_oben, bruch_unten)
    rel_wma_ln = np.log(rel_wma)

    xs = [str(y) for y in range(2001, 2021)]
    fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    ax.plot(xs, rel_wma, c='k', label='Absolut')
    ax2 = ax.twinx()
    ax2.plot(xs, rel_wma_ln, c='k', linestyle='--', label='Log.')
    ax.set_ylabel('Rel. WMA')
    ax2.set_ylabel('Rel. WMA - logarithmiert')
    ax.legend(loc=2)
    ax2.legend(loc=1)

    ax.set_xticklabels(xs, rotation=45)
    ax.grid()
    plt.suptitle(f'{ctr}, Comm. code: {what}')
    plt.subplots_adjust(left=0.08, bottom=0.18, right=0.92, top=0.9)

    plt.savefig(f'./graphics/Weltanteil_AUT_{what}.png', dpi=300)
    plt.show()

    print(rel_wma_ln)


if __name__ == "__main__":
    # what = 11101    # water
    what = 1123     # beer
    # what = 11102    # sugar drinks

    # lineplot_years_tradevalues1(spalte_of_interest='Commodity Code', what=what) #Einzeln - export
    # lineplot_years_tradevalues2(spalte_of_interest='Commodity Code', what=what) #Diffrenz Einfach
    # lineplot_years_tradevalues3(spalte_of_interest='Commodity Code', what=what) #Diffrerenz Größe Ökonomie
    # lineplot_years_tradevalues4(spalte_of_interest='Commodity Code', what=what) # Export plus Import
    # lineplot_years_tradevalues5(spalte_of_interest='Commodity Code', what=what) #Einzeln - export
    #
    # lineplot_alle_zu_welt(spalte_of_interest='Commodity Code', what=what, in_percent=False)
    barchart_alle_zu_welt(spalte_of_interest='Commodity Code', what=what, in_percent=True)

    ctr = 'CHN'
    # weltanteil(spalte_of_interest='Commodity Code', what=what, ctr=ctr)
