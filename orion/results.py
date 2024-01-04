"""Summarizing the results of the benchmark and producing leaderboard."""


import pandas as pd

DATASET_FAMILY = {
    "MSL": "NASA",
    "SMAP": "NASA",
    "UCR": "UCR",
    "YAHOOA1": "YAHOO",
    "YAHOOA2": "YAHOO",
    "YAHOOA3": "YAHOO",
    "YAHOOA4": "YAHOO",
    "artificialWithAnomaly": "NAB",
    "realAWSCloudwatch": "NAB",
    "realAdExchange": "NAB",
    "realTraffic": "NAB",
    "realTweets": "NAB"
}

DATASET_ABBREVIATION = {
    "MSL": "MSL",
    "SMAP": "SMAP",
    "UCR": "UCR",
    "YAHOOA1": "A1",
    "YAHOOA2": "A2",
    "YAHOOA3": "A3",
    "YAHOOA4": "A4",
    "artificialWithAnomaly": "Art",
    "realAWSCloudwatch": "AWS",
    "realAdExchange": "AdEx",
    "realTraffic": "Traf",
    "realTweets": "Tweets"
}


def get_f1_scores(results):
    df = results.groupby(['dataset', 'pipeline'])[['fp', 'fn', 'tp']].sum().reset_index()

    precision = df['tp'] / (df['tp'] + df['fp'])
    recall = df['tp'] / (df['tp'] + df['fn'])
    df['f1'] = 2 * (precision * recall) / (precision + recall)

    df = df.set_index(['dataset', 'pipeline'])[['f1']].unstack().T.droplevel(0)

    df.columns = [DATASET_ABBREVIATION[col] for col in df.columns]
    df.columns = pd.MultiIndex.from_tuples(list(zip(DATASET_FAMILY.values(), df.columns)))

    df['mean'] = df.mean(axis=1)
    df['std'] = df.std(axis=1)

    df.insert(0, 'Pipeline', df.index)
    df = df.reset_index(drop=True)

    return df


def get_summary_page(results):
    def get_status(x):
        return {
            "OK": 0,
            "ERROR": 1
        }[x]

    results['status'] = results['status'].apply(get_status)
    df = results.groupby(['dataset', 'pipeline'])[['fp', 'fn', 'tp']].sum().reset_index()

    precision = df['tp'] / (df['tp'] + df['fp'])
    recall = df['tp'] / (df['tp'] + df['fn'])
    df['f1'] = 2 * (precision * recall) / (precision + recall)

    summary = dict()

    # number of wins over arima
    arima_pipeline = 'arima'
    intermediate = df.set_index(['pipeline', 'dataset'])['f1'].unstack().T
    arima = intermediate.pop(arima_pipeline)

    summary['# Wins'] = (intermediate.T > arima).sum(axis=1)
    summary['# Wins'][arima_pipeline] = None

    # number of anomalies detected
    summary['# Anomalies'] = df.groupby('pipeline')[['tp', 'fp']].sum().sum(axis=1).to_dict()

    # average f1 score
    summary['Average F1 Score'] = df.groupby('pipeline')['f1'].mean().to_dict()

    # failure rate
    summary['Failure Rate'] = results.groupby(
        ['dataset', 'pipeline'])['status'].mean().unstack('pipeline').T.mean(axis=1)

    summary = pd.DataFrame(summary)
    summary.index.name = 'Pipeline'

    rank = 'Average F1 Score'
    summary.sort_values(rank, ascending=False, inplace=True)

    return summary.reset_index()


def add_sheet(df, name, writer, cell_fmt, header_fmt):
    widths = [0]
    startrow = 0
    offset = 1

    df_ = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        offset += 1
        df_.columns = df_.columns.droplevel()

    df_.to_excel(
        writer,
        sheet_name=name,
        startrow=startrow +
        offset,
        index=False,
        header=False,
        float_format="%0.4f")

    worksheet = writer.sheets[name]

    for idx, columns in enumerate(df.columns):
        column_name = columns
        if not isinstance(columns, tuple):
            columns = (columns, )

        for offset, column in enumerate(columns):
            worksheet.write(startrow + offset, idx, column, header_fmt)

            width = max(len(column), *df[column_name].astype(str).str.len()) + 1
            if len(widths) > idx:
                widths[idx] = max(widths[idx], width)
            else:
                widths.append(width)

    if isinstance(df.columns, pd.MultiIndex):
        columns = df.columns

        # horizontal
        worksheet.merge_range(0, 1, 0, 2, columns[1][0], header_fmt)
        worksheet.merge_range(0, 4, 0, 7, columns[4][0], header_fmt)
        worksheet.merge_range(0, 8, 0, 12, columns[8][0], header_fmt)

        # vertical
        worksheet.merge_range(0, 0, 1, 0, columns[0][0], header_fmt)
        worksheet.merge_range(0, 13, 1, 13, columns[13][0], header_fmt)
        worksheet.merge_range(0, 14, 1, 14, columns[14][0], header_fmt)

    for idx, width in enumerate(widths):
        worksheet.set_column(idx, idx, width + 1, cell_fmt)


def write_results(results, output, version):
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    cell_fmt = writer.book.add_format({
        "font_name": "Arial",
        "font_size": "10"
    })
    header_fmt = writer.book.add_format({
        "font_name": "Arial",
        "font_size": "10",
        "bold": True,
        "bottom": 1,
        "align": "center"
    })

    summary_page = get_summary_page(results)
    add_sheet(summary_page, version + '-Overview', writer, cell_fmt, header_fmt)
    f1_scores_page = get_f1_scores(results)
    add_sheet(f1_scores_page, version + '-F1-Scores', writer, cell_fmt, header_fmt)

    writer.save()
