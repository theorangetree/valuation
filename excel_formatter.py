"""Export DCF valuation model to formatted Excel file"""
import datetime
import pandas as pd
import numpy as np

def export_dcf(tuple_list): # [(ticker, name, dcf_output, final_output)]
    """Collect DCF valuation outputs and export DCF valuation model to formatted Excel file

    Valuation outputs must be collected as a list of tuples; one tuple per ticker, with each containing:
        # dcf_output   -- DataFrame of DCF model from base year through terminal year
        # final_output -- DataFrame containing final valuation steps
        # ticker       -- company ticker
        # name         -- company name
        # ratio        -- company value-to-price ratio
        # assumptions  -- DCF model inputs (including assumptions)
    """
    with pd.ExcelWriter('DCF valuations.xlsx', date_format='YYYY-MM-DD') as writer:
        for dcf_output, final_output, ticker, name, ratio, assumptions in tuple_list:

            # Insert blank columns for aesthetics
            dcf_output.insert(dcf_output.shape[1]-1,'',np.nan,allow_duplicates=True)
            dcf_output.insert(1                    ,'',np.nan,allow_duplicates=True)

            # Size and starting cell for DCF output (top-left corner)
            start_row   = 3
            start_col   = 0
            dcf_height  = dcf_output.shape[0]
            dcf_width   = dcf_output.shape[1]

            # Assign workbook and worksheet objects
            workbook     = writer.book
            output_sheet = workbook.add_worksheet(ticker)

            # Write title
            title_format     = workbook.add_format({'bold':True, 'size':16, 'bottom':6})
            double_underline = workbook.add_format({'bottom':6})
            for col in range(start_col + 1, start_col + 1 + dcf_width):
                output_sheet.write(0, col, None, double_underline)
            output_sheet.write(0, 0, f'Valuation of {name} on {datetime.date.today()}',title_format)

            # Identify format orders for columns and row
            dcf_col_order   = ['hc', 'blank', 'left'] + ['pc']*(dcf_width-6) + ['right', 'blank', 'hc']
            dcf_row_order   = ['percent', 'bold']*2 + ['number']*2 + ['percent', 'bold', 'percent', 'number', 'bold'] + ['percent']*2 + ['bold']
            final_row_order = ['number']*2 + ['bold'] + ['number']*4 + ['bold'] + ['number']*2 + ['final_dollar_top', 'final_dollar_mid', 'final_percent']

            # Primary cell colors
            pc1 = '#333F4F' # Dark color for headers
            pc2 = '#EDEDED' # Light color for indices
            pc3 = '#E3F1F9' # Light color for data
            # Highlight cell colors
            hc1 = '#2675CC' # Dark color for headers
            hc2 = '#D3E3F5' # Light color for data
            # Final result cell colors
            fc1 = '#222B35' # Very dark color for indices
            fc2 = '#1F4D82' # Dark color for data

            # Write columns headers
            hc_header       = workbook.add_format({'num_format':'YYYY-MM-DD', 'bold':True, 'bg_color':hc1, 'align':'center', 'border':1, 'color':'#FFFFFF'})
            hc_header_top   = workbook.add_format({'num_format':'YYYY-MM-DD', 'bold':True, 'bg_color':hc1, 'align':'center', 'border':1, 'color':'#FFFFFF', 'bottom_color':'#FFFFFF'})
            hc_header_bot   = workbook.add_format({'num_format':'YYYY-MM-DD', 'bold':True, 'bg_color':hc1, 'align':'center', 'border':1, 'color':'#FFFFFF', 'top_color'   :'#FFFFFF'})
            pc_header       = workbook.add_format({'num_format':'YYYY-MM-DD', 'bold':True, 'bg_color':pc1, 'align':'center', 'border':1, 'color':'#FFFFFF', 'left_color'  :'#FFFFFF', 'right_color':'#FFFFFF'})
            pc_header_left  = workbook.add_format({'num_format':'YYYY-MM-DD', 'bold':True, 'bg_color':pc1, 'align':'center', 'border':1, 'color':'#FFFFFF', 'right_color' :'#FFFFFF'})
            pc_header_right = workbook.add_format({'num_format':'YYYY-MM-DD', 'bold':True, 'bg_color':pc1, 'align':'center', 'border':1, 'color':'#FFFFFF', 'left_color'  :'#FFFFFF'})
            blank_format    = workbook.add_format({})
            header_format_dict = {'hc': hc_header_bot, 'blank': blank_format, 'pc': pc_header, 'left': pc_header_left, 'right': pc_header_right}

            output_sheet.write(start_row                 , start_col            , 'Year Ended'     , pc_header)
            output_sheet.write(start_row - 1             , start_col + 1        , 'Base Year'      , hc_header_top)
            output_sheet.write(start_row - 1             , start_col + dcf_width, 'Terminal Year'  , hc_header_top)
            output_sheet.write(start_row + dcf_height + 2, start_col + 1        , 'Final Valuation', hc_header)

            for col_index, value in enumerate(dcf_output.columns):
                cell_format = header_format_dict[dcf_col_order[col_index]]
                output_sheet.write(start_row, start_col + col_index + 1, value, cell_format)

            # Write indices
            index_format        = workbook.add_format({'bg_color':pc2, 'left':1, 'right':1, 'bottom':1, 'left_color':'#000000', 'right_color':'#000000', 'bottom_color':'#FFFFFF'})
            index_bold_format   = workbook.add_format({'bg_color':pc2, 'left':1, 'right':1, 'bottom':1, 'left_color':'#000000', 'right_color':'#000000', 'bottom_color':'#000000', 'bold':True})
            index_dark_grey_top = workbook.add_format({'bg_color':fc1, 'border':1, 'bold':True, 'color':'#FFFFFF', 'bottom_color':'#FFFFFF'})
            index_dark_grey_mid = workbook.add_format({'bg_color':fc1, 'border':1, 'bold':True, 'color':'#FFFFFF', 'bottom_color':'#FFFFFF', 'top_color':'#FFFFFF'})
            index_dark_grey_bot = workbook.add_format({'bg_color':fc1, 'border':1, 'bold':True, 'color':'#FFFFFF', 'top_color':'#FFFFFF'})
            index_line          = workbook.add_format({'bottom':1, 'bottom_color':'#000000'})
            index_format_dict = {'number': index_format, 'bold': index_bold_format, 'percent': index_format}
            final_index_dict  = {'number': index_format, 'bold': index_bold_format, 'final_dollar_top': index_dark_grey_top, 'final_dollar_mid': index_dark_grey_mid, 'final_percent': index_dark_grey_bot}

            for row_index, value in enumerate(dcf_output.index):
                cell_format = index_format_dict[dcf_row_order[row_index]]
                output_sheet.write(start_row + row_index + 1, start_col, value, cell_format)

            for row_index, value in enumerate(final_output.index):
                cell_format = final_index_dict[final_row_order[row_index]]
                output_sheet.write(start_row + dcf_height + row_index + 3, start_col, value, cell_format)

            output_sheet.write(start_row + dcf_height + 2, start_col, None, index_line)

            # Data frame cell formats
            hc_percent    = workbook.add_format({'num_format':'0.00%', 'bg_color':hc2, 'align':'center', 'left':1, 'right':1, 'bottom':1, 'left_color':'#000000', 'right_color':'#000000', 'bottom_color':'#FFFFFF'})
            hc_number     = workbook.add_format({'num_format':'#,##0', 'bg_color':hc2, 'align':'center', 'left':1, 'right':1, 'bottom':1, 'left_color':'#000000', 'right_color':'#000000', 'bottom_color':'#FFFFFF'})
            hc_bold       = workbook.add_format({'num_format':'#,##0', 'bg_color':hc2, 'align':'center', 'left':1, 'right':1, 'bottom':1, 'left_color':'#000000', 'right_color':'#000000', 'bottom_color':'#000000', 'bold':True})
            pc_percent    = workbook.add_format({'num_format':'0.00%', 'bg_color':pc3, 'align':'center', 'left':1, 'right':1, 'bottom':1, 'left_color':'#FFFFFF', 'right_color':'#FFFFFF', 'bottom_color':'#FFFFFF'})
            pc_number     = workbook.add_format({'num_format':'#,##0', 'bg_color':pc3, 'align':'center', 'left':1, 'right':1, 'bottom':1, 'left_color':'#FFFFFF', 'right_color':'#FFFFFF', 'bottom_color':'#FFFFFF'})
            pc_bold       = workbook.add_format({'num_format':'#,##0', 'bg_color':pc3, 'align':'center', 'left':1, 'right':1, 'bottom':1, 'left_color':'#FFFFFF', 'right_color':'#FFFFFF', 'bottom_color':'#000000', 'bold':True})
            left_percent  = workbook.add_format({'num_format':'0.00%', 'bg_color':pc3, 'align':'center', 'left':1, 'right':1, 'bottom':1, 'left_color':'#000000', 'right_color':'#FFFFFF', 'bottom_color':'#FFFFFF'})
            left_number   = workbook.add_format({'num_format':'#,##0', 'bg_color':pc3, 'align':'center', 'left':1, 'right':1, 'bottom':1, 'left_color':'#000000', 'right_color':'#FFFFFF', 'bottom_color':'#FFFFFF'})
            left_bold     = workbook.add_format({'num_format':'#,##0', 'bg_color':pc3, 'align':'center', 'left':1, 'right':1, 'bottom':1, 'left_color':'#000000', 'right_color':'#FFFFFF', 'bottom_color':'#000000', 'bold':True})
            right_percent = workbook.add_format({'num_format':'0.00%', 'bg_color':pc3, 'align':'center', 'left':1, 'right':1, 'bottom':1, 'left_color':'#FFFFFF', 'right_color':'#000000', 'bottom_color':'#FFFFFF'})
            right_number  = workbook.add_format({'num_format':'#,##0', 'bg_color':pc3, 'align':'center', 'left':1, 'right':1, 'bottom':1, 'left_color':'#FFFFFF', 'right_color':'#000000', 'bottom_color':'#FFFFFF'})
            right_bold    = workbook.add_format({'num_format':'#,##0', 'bg_color':pc3, 'align':'center', 'left':1, 'right':1, 'bottom':1, 'left_color':'#FFFFFF', 'right_color':'#000000', 'bottom_color':'#000000', 'bold':True})
            final_dollar_top = workbook.add_format({'num_format':'$#,##0.00', 'bg_color':fc2, 'border':1, 'align':'center', 'color':'#FFFFFF', 'bold':True, 'bottom_color':'#FFFFFF'})
            final_dollar_mid = workbook.add_format({'num_format':'$#,##0.00', 'bg_color':fc2, 'border':1, 'align':'center', 'color':'#FFFFFF', 'bold':True, 'bottom_color':'#FFFFFF', 'top_color':'#FFFFFF'})
            final_percent    = workbook.add_format({'num_format':'0.00%'    , 'bg_color':fc2, 'border':1, 'align':'center', 'color':'#FFFFFF', 'bold':True, 'top_color'   :'#FFFFFF'})

            # Dictionary of all data formats
            data_format_dict  = {'hc'    :{'percent': hc_percent, 'number': hc_number, 'bold': hc_bold},
                                 'left'  :{'percent': left_percent  , 'number': left_number  , 'bold': left_bold  },
                                 'pc'    :{'percent': pc_percent  , 'number': pc_number  , 'bold': pc_bold  },
                                 'right' :{'percent': right_percent , 'number': right_number , 'bold': right_bold }}
            final_format_dict = {'number': hc_number, 'bold': hc_bold, 'final_dollar_top': final_dollar_top, 'final_dollar_mid': final_dollar_mid, 'final_percent': final_percent}

            # Write data
            for col_index, col_format in enumerate(dcf_col_order):
                for row_index, (row_format, value) in enumerate(zip(dcf_row_order, dcf_output.iloc[:,col_index])):
                    if col_format == 'blank':
                        cell_format = blank_format
                        value = None
                    else:
                        cell_format = data_format_dict[col_format][row_format]
                    try:
                        output_sheet.write(start_row + row_index + 1, start_col + col_index + 1, value, cell_format)
                    except TypeError:
                        output_sheet.write(start_row + row_index + 1, start_col + col_index + 1, '--' , cell_format)

            for row_index, (row_format, value) in enumerate(zip(final_row_order, final_output.iloc[:,0])):
                cell_format = final_format_dict[row_format]
                output_sheet.write(start_row + dcf_height + row_index + 3, start_col + 1, value, cell_format)

            # Row heights
            output_sheet.set_row(0, 25)
            output_sheet.set_row(1, 22)

            # Column widths
            output_sheet.set_column('A:A',27)
            output_sheet.set_column('B:B',16)
            output_sheet.set_column('C:C',5)
            output_sheet.set_column('D:M',14)
            output_sheet.set_column('N:N',5)
            output_sheet.set_column('O:O',16)

            # Set tab color based on value-to-price ratio
            lower_bound = 1/3
            upper_bound = 3

            if ratio <= 1:
                zero_to_one_scale = max(0, 0.5 * (1 + 1/(lower_bound - 1))  +  ratio / (2*(1 - lower_bound))) # ratio == lower_bound returns 0
            else:                                                                                             # ratio == 1           returns 0.5
                zero_to_one_scale = min(1, 0.5 * (1 - 1/(upper_bound - 1))  +  ratio / (2*(upper_bound - 1))) # ratio == upper_bound returns 1

            # Color scale (red = low value ratio; yellow = 1:1 ratio; green = high value ratio)
            red   = int(min(225, 450 - 450 * zero_to_one_scale))
            green = int(min(225, 0   + 450 * zero_to_one_scale))
            hex_code = f'#{red:02X}{green:02X}{0:02X}'
            output_sheet.set_tab_color(hex_code)

            # Freeze panes and hide gridlines
            output_sheet.freeze_panes(4,1)
            output_sheet.hide_gridlines(2)
