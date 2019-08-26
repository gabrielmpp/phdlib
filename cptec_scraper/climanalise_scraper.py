from bs4 import BeautifulSoup
import urllib
import pandas as pd
import requests
import warnings
from tqdm import tqdm  # progress bar
import re
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import datetime

def exists(URL):
    r = requests.head(URL)
    return r.status_code == requests.codes.ok

def get_last_month(month, year):
    last_month = month - 1
    if last_month == 0:
        last_month = 12
        year = year - 1
    elif last_month < 0 or last_month > 13:
        raise ValueError("Invalid month!")
    return last_month, year

def like(string=None, mode='all'):
    """
    Return a compiled regular expression that matches the given
    string with any prefix and postfix, e.g. if string = "hello",
    the returned regex matches r".*hello.*"
    """
    string_ = string
    if not isinstance(string_, str):
        string_ = str(string_)
    if mode == 'all':
        MATCH_ALL = r'.*'
        regex = MATCH_ALL + re.escape(string_) + MATCH_ALL
    if mode == 'dia_a_dia':
        regex = r'\d{1,2} (.* )?[a,e,até]( .*)? \d{1,2} $'
    return re.compile(regex, flags=re.DOTALL)


class scrapper:
    """
    Web scrapper for Climanalise reports
    """

    @staticmethod
    def scrape(init_date="2005/01/01", end_date="2014/12/31", keyword="ZCAS", div_id='sub12'):
        full_dates = pd.date_range(init_date, end_date, freq="M")
        dates = [pd.to_datetime(date).strftime("%m%y") for date in full_dates]
        outdict = {}
        for idx, date in enumerate(tqdm(dates)):
            urlstr = f'http://climanalise.cptec.inpe.br/~rclimanl/boletim/index{date}.shtml'
            if exists(urlstr):
                url = urllib.request.urlopen(urlstr)
                html_content = url.read()  # getting html content
                soup = BeautifulSoup(html_content)
                paragraph = soup.find(id=div_id)  # find div where zcas info is contained
                if paragraph.find_all(text=like(keyword)):
                    outdict[full_dates[idx]] = {}
                    outdict[full_dates[idx]]['text'] = paragraph
                    days = re.findall(r" (?<!Volume )\d{2}(?! horas| semanas)(?=[ ,])", paragraph.contents.__str__())
                    if len(days) % 2 != 0: days = days[:-1] # dropping last day cause it refers to next month
                    print(paragraph)
                    print(days)
                    if len(days) >= 2:
                        outdict[full_dates[idx]]['days'] = days  # TODO CHECK IF DAY2>1 ELSE MONTH = MONTH-1
                        outdict[full_dates[idx]]['month_delta'] = 1 if days[0] > days[1] else 0
                        outdict[full_dates[idx]]['num_of_cases'] = len(days)/2
                    else:
                        outdict[full_dates[idx]] = {}
                        outdict[full_dates[idx]]['text'] = None
                        outdict[full_dates[idx]]['days'] = []
                        outdict[full_dates[idx]]['num_of_cases'] = 0
                else:
                    outdict[full_dates[idx]] = {}
                    outdict[full_dates[idx]]['text'] = None
                    outdict[full_dates[idx]]['days'] = []
                    outdict[full_dates[idx]]['num_of_cases'] = 0

            else:
                warnings.warn(f"Url {urlstr} does not exist.", Warning)

        data = pd.DataFrame.from_dict(outdict, orient='index')
        data = data[['num_of_cases', 'days','month_delta']]
        array = data.to_xarray()
        return array

    @staticmethod
    def format_array(array):
        full_dates = []
        for idx, days in enumerate(array.days.values):

            ks = [[x, x + 1] for x in range(0, len(days), 2)] #pairwise indexes
            for jdx, k in enumerate(ks):
                onset_day, demise_day = [days[int(i)] for i in k]
                ## TODO setting up a datetime sequence between onset and demise
                month = pd.to_datetime(array.index.values[idx]).month
                #month = month - array.month_delta.values[idx] if jdx == 0 else month
                year = pd.to_datetime(array.index.values[idx]).year
                onset = datetime.date(day=int(onset_day), month=month, year=year)
                demise = datetime.date(day=int(demise_day), month=month, year=year)
                if onset > demise:
                    month, year = get_last_month(onset.month, onset.year)
                    demise = datetime.date(day=int(onset_day), month=month, year=year)

                full_dates = full_dates + pd.date_range(start=onset, end=demise).to_list()
        time = pd.date_range(full_dates[0], full_dates[-1], freq="D").to_pydatetime()
        formatted_array = xr.DataArray(np.ones(len(time)), dims=['time'], coords={'time': time})
        mask = [pd.Timestamp(time) in full_dates for time in formatted_array.time.values]
        formatted_array = formatted_array.where(mask, 0)
        return formatted_array



if __name__ == '__main__':
    array = scrapper.scrape()
    array = scrapper.format_array(array)
    data = pd.DataFrame.from_dict(paragraphs, orient='index')
    data = data['num_of_cases']
    array = data.to_xarray()

    data['2010-01-01']
    data.plot()

    print('done')