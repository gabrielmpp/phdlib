from bs4 import BeautifulSoup
import urllib
import pandas as pd
import requests
import warnings
from tqdm import tqdm  # progress bar


def exists(URL):
    r = requests.head(URL)
    return r.status_code == requests.codes.ok


class scrapper:
    """
    Web scrapper for Climanalise reports
    """

    @staticmethod
    def scrape(init_date="2005/01/01", end_date="2014/12/31", keyword="ZCAS", div_id='sub12'):
        dates = pd.date_range(init_date, end_date, freq="M")
        dates = [pd.to_datetime(date).strftime("%m%y") for date in dates]
        outdict = {}
        for date in tqdm(dates):
            urlstr = f'http://climanalise.cptec.inpe.br/~rclimanl/boletim/index{date}.shtml'
            if exists(urlstr):
                url = urllib.request.urlopen(urlstr)
                html_content = url.read()
                soup = BeautifulSoup(html_content)
                try:
                    paragraph = soup.find(id=div_id)
                    if keyword in paragraph:
                        outdict['date'] = paragraph
                except:
                    warnings.warn(f"id {div_id} not found", Warning)

            else:
                warnings.warn(f"Url {urlstr} does not exist.", Warning)
        return outdict


if __name__ == '__main__':
    paragraphs = scrapper.scrape()
    print('done')