from sight import Sightseer
from proc import DataAnnotator

da = DataAnnotator(['racoon'])

xmlpath = "./test_data/"
csvpath = "./test_data/csv/"
da.xml_to_csv(xmlpath, csvpath)