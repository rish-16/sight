from sight import Sightseer
from proc import DataAnnotator

da = DataAnnotator(['racoon'])

xmlpath = "./test_data/xml/"
csvfile = "./test_data/csv/test.csv"
da.xml_to_csv(xmlpath, csvfile)