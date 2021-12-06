#!/usr/bin/env python3

import click
import glob
from pathlib import Path
import pandas as pd
from anapi.pandas_utils import (
    write_dataframe_to_tsv, read_tsv_into_dataframe, sort_df, filter_by_col_val, fill_nan
)

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.version_option(version='0.1')
@click.option('-t', '--table_path',
              required=True,
              help="Path to taxa_tables.")
@click.option('-o', '--output_folder',
              required=False,
              help="Output file for results table")
@click.option('-m', '--map_path',
              required=False,
              help="Full path to mapping file")
@click.option('-ao', '--analysis_option',
              required=True,
              help="Analysis option to execute")
@click.option('-c', '--column',
              required=False,
              help="Full path to mapping file")
@click.option('-a', '--analysis_name',
              required=False,
              help="Full path to mapping file")
def table_analysis(table_path, output_folder, map_path, analysis_option, column, analysis_name):
    """
    Generic script for any sort of table-based calculations or transformations.
    taxa_table_path: path to folder containing taxa tables
    output_folder: path for output files to be written to.
    map_path: path to mapping file
    """
    # Create output folder
    output_dir = Path(output_folder)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    table_path = Path(table_path) / '*.tsv'
    table_files = glob.glob(str(table_path))

    for i, tsv_file in enumerate(table_files):
        taxa_df = read_tsv_into_dataframe(tsv_file)

        if analysis_option == 'sort':
            taxa_df[column] = taxa_df[column].astype(float)
            taxa_df = sort_df(taxa_df, column=column, ascending=False)
        elif analysis_option == 'wide-to-long':
            # success
            df1 = pd.melt(taxa_df, id_vars=['PID'],
                              value_vars=['Live-cm2', 'Dead-cm2', 'Total-cm2'], var_name='cm2',
                          value_name='cm2-val')
            df1['state'] = df1['cm2'].str.replace('-cm2', '')

            df2 = pd.melt(taxa_df, id_vars=['PID'],
                              value_vars=['#Live', '#Dead', '#Total'], var_name='number', value_name='number-val')
            df2['state'] = df2['number'].str.replace('#', '')


            taxa_df['Total%'] = 100
            df3 = pd.melt(taxa_df, id_vars=['PID'],
                              value_vars=['Live%', 'Dead%', 'Total%'], var_name='percent', value_name='percent-val')
            df3['state'] = df3['percent'].str.replace('%', '')

            final_df = df1.merge(df2, on=['PID', 'state'], how='outer')
            final_df = final_df.merge(df3, on=['PID', 'state'], how='outer')

            final_df['order'] = final_df['state'].str.replace('Dead', 'ZDead')
            final_df['order2'] = final_df['order'].str.replace('Live', 'YLive')


            final_df.sort_values(['PID', 'order2'], ascending=True, inplace=True)
            #final_df = sort_df(final_df, column='PID', ascending=True)
            final_df.drop('state', 1, inplace=True)
            final_df.drop('order', 1, inplace=True)
            final_df.drop('order2', 1, inplace=True)

            table_name = str(Path(tsv_file).stem)
            write_dataframe_to_tsv(final_df, f'{table_name}.tsv', output_folder)

        elif analysis_option == 'long-to-wide':
            if 'Metadata' in str(tsv_file):
            # drop unneeded columns
            # combine taxonomic columns
                cols = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']

                # redo with rel counts
                if 'relab' in str(tsv_file):

                    drop_cols = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species', 'feature_id',
                                'sample_name', 'counts', 'well_id', 'plate_id', 'timepoint', 'group', 'sample_weight_mg',
                                'sample_date', 'sample_type', 'project_name', 'investigator', 'mice',
                                'nucleic_acid_after_extraction_ng_ul', 'gut_trypsin_activity_mu_ml', 'abs_counts']
                elif 'absab' in str(tsv_file):
                    drop_cols = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species', 'feature_id',
                                'sample_name', 'counts', 'well_id', 'plate_id', 'timepoint', 'group', 'sample_weight_mg',
                                'sample_date', 'sample_type', 'project_name', 'investigator', 'mice',
                                'nucleic_acid_after_extraction_ng_ul', 'gut_trypsin_activity_mu_ml', 'rel_counts']


                EAE_df = filter_by_col_val(taxa_df, ["mice:'EAE'"], method='filter_stmt')
                EAE_df['taxa'] = EAE_df[cols].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
                EAE_df = EAE_df.drop(columns=drop_cols)
                EAE_df = pd.pivot_table(EAE_df, index='mouse_id', columns='taxa').T
                EAE_df.reset_index(inplace=True)
                EAE_df = EAE_df.drop(columns='level_0')
                EAE_df['taxa'] = EAE_df['taxa'].str.replace('_nan', '')
                table_name = str(Path(tsv_file).stem)
                table_name = table_name.split('_')[0] + '_' + table_name.split('_')[-1]

                write_dataframe_to_tsv(EAE_df, f'{table_name}_EAE.tsv', output_folder)

                Naive_df = filter_by_col_val(taxa_df, ["mice:'Naive'"], method='filter_stmt')
                Naive_df['taxa'] = Naive_df[cols].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
                Naive_df = Naive_df.drop(columns=drop_cols)
                Naive_df = pd.pivot_table(Naive_df, index='mouse_id', columns='taxa').T
                Naive_df.reset_index(inplace=True)
                Naive_df = Naive_df.drop(columns='level_1')
                Naive_df['taxa'] = Naive_df['taxa'].str.replace('_nan', '')
                table_name = str(Path(tsv_file).stem)
                table_name = table_name.split('_')[0] + '_' + table_name.split('_')[-1]

                write_dataframe_to_tsv(Naive_df, f'{table_name}_Naive.tsv', output_folder)

                taxa_df['taxa'] = taxa_df[cols].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
                taxa_df = taxa_df.drop(columns=drop_cols)
                taxa_df = pd.pivot_table(taxa_df, index='mouse_id', columns='taxa').T
                taxa_df.reset_index(inplace=True)
                taxa_df = taxa_df.drop(columns='level_0')
                taxa_df['taxa'] = taxa_df['taxa'].str.replace('_nan', '')
                table_name = str(Path(tsv_file).stem)
                table_name = table_name.split('_')[0] + '_' + table_name.split('_')[-1]

                write_dataframe_to_tsv(taxa_df, f'{table_name}_both.tsv', output_folder)


            elif 'Significant' in str(tsv_file):
                drop_cols = ['ID', 'pvalue', 'padj']
                taxa_df = taxa_df.drop(columns=drop_cols)
                taxa_df.set_index('Gene.name', inplace=True)
                taxa_df.reset_index(inplace=True)
                taxa_df.columns = taxa_df.columns.str.replace('-resub', '')

                table_name = str(Path(tsv_file).stem)
                # table_name = table_name.split('_')[0]
                write_dataframe_to_tsv(taxa_df, f'{table_name}_genes.tsv', output_folder)


            elif 'Bile' in str(tsv_file):
                taxa_df['Result'] = taxa_df['Result'].astype(float)
                drop_cols = ['Client Matrix', 'Group Number', 'Group', 'Unit', 'Experiment']

                EAE_df = filter_by_col_val(taxa_df.copy(), ['Group:EAE'], method='filter_like')
                EAE_df = EAE_df.drop(columns=drop_cols, errors='ignore')
                EAE_df = pd.pivot_table(EAE_df, index='Sample_ID', columns='Analyte')
                EAE_df = EAE_df.T
                EAE_df.reset_index(inplace=True)
                EAE_df = EAE_df.drop(columns=EAE_df.columns[0])
                table_name = str(Path(tsv_file).stem)
                table_name = table_name.split('_')[0]

                write_dataframe_to_tsv(EAE_df, f'{table_name}_EAE.tsv', output_folder)

                Naive_df = filter_by_col_val(taxa_df.copy(), ['Group:Naive'], method='filter_like')
                Naive_df = Naive_df.drop(columns=drop_cols, errors='ignore')
                Naive_df = pd.pivot_table(Naive_df, index='Sample_ID', columns='Analyte')
                Naive_df = Naive_df.T
                Naive_df.reset_index(inplace=True)
                Naive_df = Naive_df.drop(columns=Naive_df.columns[0])
                table_name = str(Path(tsv_file).stem)
                table_name = table_name.split('_')[0]

                write_dataframe_to_tsv(Naive_df, f'{table_name}_Naive.tsv', output_folder)

                taxa_df = taxa_df.drop(columns=drop_cols, errors='ignore')
                taxa_df = pd.pivot_table(taxa_df, index='Sample_ID', columns='Analyte')
                taxa_df = taxa_df.T
                taxa_df.reset_index(inplace=True)
                taxa_df = taxa_df.drop(columns=taxa_df.columns[0])
                table_name = str(Path(tsv_file).stem)
                table_name = table_name.split('_')[0]
                write_dataframe_to_tsv(taxa_df, f'{table_name}_both.tsv', output_folder)


            elif 'EAEscore' in str(tsv_file):
                # taxa_df['Result'] = pd.to_numeric(taxa_df['Result'].astype(float)
                # taxa_df = pd.pivot_table(taxa_df, index='Mice_ID', columns='Group')
                taxa_df = taxa_df.drop(columns='Group')
                taxa_df.set_index(taxa_df.columns[0], inplace=True)

                taxa_df = taxa_df.T
                taxa_df.reset_index(inplace=True)
                # taxa_df = taxa_df.drop(columns='level_0')

                table_name = str(Path(tsv_file).stem)
                table_name = table_name.split('_')[0]
                write_dataframe_to_tsv(taxa_df, f'{table_name}.tsv', output_folder)

            else:
                drop_cols = ['Client Matrix', 'Group Number', 'Group', 'Unit', 'Experiment']
                taxa_df = taxa_df.drop(columns=drop_cols, errors='ignore')
                # taxa_df.set_index(taxa_df.columns[0], inplace=True)

                taxa_df['Result'] = taxa_df['Result'].astype(float)
                # taxa_df['Result'] = pd.to_numeric(taxa_df['Result'].astype(float)

                taxa_df = pd.pivot_table(taxa_df, index='Sample_ID', columns='Analyte')
                taxa_df = taxa_df.T
                taxa_df.reset_index(inplace=True)
                taxa_df = taxa_df.drop(columns='level_0')
                # taxa_df.set_index(taxa_df.columns[0], inplace=True)
                # taxa_df = taxa_df.T
                # taxa_df.reset_index(inplace=True)

                table_name = str(Path(tsv_file).stem)
                table_name = table_name.replace('Trp_metabolites_Ampicillin_Vancomycin_EAE_8.31.20', 'Trp-EAE')
                table_name = table_name.replace('Trp_metabolites_VancoAmp_Naive_mice_8.31.20', 'Trp-Naive')
                write_dataframe_to_tsv(taxa_df, f'{table_name}.tsv', output_folder)


            # final_table.columns = final_table.columns.str.replace('value', 'abundance')

if __name__ == '__main__':
    table_analysis()
