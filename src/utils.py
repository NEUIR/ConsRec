import random
import numpy as np
import torch
import sys
import logging


logger = logging.getLogger(__name__)


def init_logger(filename=None):

    handlers = [logging.StreamHandler(sys.stdout)]
    if filename is not None:
        handlers.append(logging.FileHandler(filename=filename))
    logging.basicConfig(
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
        handlers=handlers,
    )
    logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    return logger


def early_stopping(value, best, cur_step, max_step):

    stop_flag = False
    update_flag = False

    if value > best:
        cur_step = 0
        best = value
        update_flag = True
    else:
        cur_step += 1
        if cur_step > max_step:
            stop_flag = True

    return best, cur_step, stop_flag, update_flag


def set_randomseed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


dataset2class = {
    "onion": "Music4AllOnion",
    "ml-100k": "ML100KDataset",
    "ml-1m": "ML1MDataset",
    "ml-10m": "ML10MDataset",
    "ml-20m": "ML20MDataset",
    "avazu": "AVAZUDataset",
    "adult": "ADULTDataset",
    "tmall": "TMALLDataset",
    "netflix": "NETFLIXDataset",
    "criteo": "CRITEODataset",
    "foursquare": "FOURSQUAREDataset",
    "diginetica": "DIGINETICADataset",
    "anime": "ANIMEDataset",
    "epinions": "EPINIONSDataset",
    "gowalla": "GOWALLADataset",
    "lfm1b": "LFM1bDataset",
    "book-crossing": "BOOKCROSSINGDataset",
    "ipinyou": "IPINYOUDataset",
    "steam": "STEAMDataset",
    "phishing-website": "PHISHINGWEBDataset",
    "pinterest": "PINTERESTDataset",
    "jester": "JESTERDataset",
    "douban": "DOUBANDataset",
    "kdd2010-algebra2006": "KDD2010Algebra2006Dataset",
    "kdd2010-algebra2008": "KDD2010Algebra2008Dataset",
    "kdd2010-bridge-to-algebra2006": "KDD2010BridgeToAlgebra2006Dataset",
    "amazon_apps_for_android": "AmazonAppsForAndroidDataset",
    "amazon_beauty": "AmazonBeautyDataset",
    "amazon_tools_and_home_improvement": "AmazonToolsAndHomeImprovementDataset",
    "amazon_books": "AmazonBooksDataset",
    "amazon_instant_video": "AmazonInstantVideoDataset",
    "amazon_digital_music": "AmazonDigitalMusicDataset",
    "amazon_movies_and_tv": "AmazonMoviesAndTVDataset",
    "amazon_automotive": "AmazonAutomotiveDataset",
    "amazon_baby": "AmazonBabyDataset",
    "amazon_clothing_shoes_and_jewelry": "AmazonClothingShoesAndJewelryDataset",
    "amazon_cell_phones_and_accessories": "AmazonCellPhonesAndAccessoriesDataset",
    "amazon_patio_lawn_and_garden": "AmazonPatioLawnAndGardenDataset",
    "amazon_kindle_store": "AmazonKindleStoreDataset",
    "amazon_home_and_kitchen": "AmazonHomeAndKitchenDataset",
    "amazon_grocery_and_gourmet_food": "AmazonGroceryAndGourmetFoodDataset",
    "amazon_health_and_personal_care": "AmazonHealthAndPersonalCareDataset",
    "amazon_pet_supplies": "AmazonPetSuppliesDataset",
    "amazon_sports_and_outdoors": "AmazonSportsAndOutdoorsDataset",
    "amazon_toys_and_games": "AmazonToysAndGamesDataset",
    "amazon_electronics": "AmazonElectronicsDataset",
    "amazon_office_products": "AmazonOfficeProductsDataset",
    "amazon_video_games": "AmazonVideoGamesDataset",
    "amazon_musical_instruments": "AmazonMusicalInstrumentsDataset",
    "yelp": "YELPDataset",
    "lastfm": "LASTFMDataset",
    "yoochoose": "YOOCHOOSEDataset",
    "yahoo-music": "YAHOOMUSICDataset",
    "ta-feng": "TAFENGDataset",
    "retailrocket": "RETAILROCKETDataset",
    "mind_large_train": "MINDLargeTrainDataset",
    "mind_large_dev": "MINDLargeDevDataset",
    "mind_small_train": "MINDSmallTrainDataset",
    "mind_small_dev": "MINDSmallDevDataset",
    "cosmetics": "CosmeticsDataset",
}

click_dataset = {
    "YOOCHOOSEDataset",
    "RETAILROCKETDataset",
    "TMALLDataset",
    "IPINYOUDataset",
    "TAFENGDataset",
    "LFM1bDataset",
    "GOWALLADataset",
    "DIGINETICADataset",
    "FOURSQUAREDataset",
    "STEAMDataset",
}

multiple_dataset = {
    "Music4AllOnion",
    "YOOCHOOSEDataset",
    "RETAILROCKETDataset",
    "TAFENGDataset",
    "TMALLDataset",
    "IPINYOUDataset",
    "LFM1bDataset",
}

multiple_item_features = {
    "Music4AllOnion",
}
