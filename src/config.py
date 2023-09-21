import os


config = {
    "binance_keys": {
            "binanceaccount1": {"API_KEY": "", "SECRET_KEY": ""},
            "binanceaccount2": {"API_KEY": "", "SECRET_KEY": ""},
            # Examaple using environment variable
            "binanceaccount3": {"API_KEY": os.environ.get("BINANCE_API_KEY_3"), 
                                "SECRET_KEY": os.environ.get("BINANCE_SECRET_KEY_3")}
    },
    "binance_test_keys": {
            "binancetest1": {"API_KEY": "", "SECRET_KEY": ""},
            "binancetest2": {"API_KEY": "", "SECRET_KEY": ""}
    },
    "bybit_keys": {
            "bybitaccount1": {"API_KEY": "", "SECRET_KEY": ""},
            "bybitaccount2": {"API_KEY": "", "SECRET_KEY": ""}
    },
    "bybit_test_keys": {
            "bybittest1": {"API_KEY": "", "SECRET_KEY": ""},
            "bybittest2": {"API_KEY": "", "SECRET_KEY": ""}
    },
    "bitmex_keys": {
            "bitmexaccount1": {"API_KEY": "", "SECRET_KEY": ""},
            "bitmexaccount2": {"API_KEY": "", "SECRET_KEY": ""}
    },
    "bitmex_test_keys": {
            "bitmextest1":{"API_KEY": "", "SECRET_KEY": ""},
            "bitmextest2": {"API_KEY": "", "SECRET_KEY": ""}
    },
    "ftx_keys": {
            "ftxaccount1": {"API_KEY": "", "SECRET_KEY": ""},
            "ftxaccount2": {"API_KEY": "", "SECRET_KEY": ""},                    
    },  
    "line_apikey": {"API_KEY": ""},
    "discord_webhooks": {
            "binanceaccount1": "",
            "binanceaccount2": "",
            "bybitaccount1": "",
            "bybitaccount2": ""
    },
    "healthchecks.io": {
                    "binanceaccount1": {
                            "websocket_heartbeat": "",
                            "listenkey_heartbeat": ""
                    },
                    "bybitaccount1": {
                            "websocket_heartbeat": "",
                            "listenkey_heartbeat": ""
                    }
    },
    "influx_db": {
                "binanceaccount1": {
                                "url" : "",
                                "org": "",
                                "token": "",
                                "bucket": ""
                },
                "bybitaccount1": {
                                "url" : "",
                                "org": "",
                                "token": "",
                                "bucket": ""
                }
    },
    # To use Args profiles, add them here and run by using the flag --profile <your profile string>
    "args_profile": {"binanceaccount1_Sample_ethusdt": {"--test": True,
                                                        "--stub": False,
                                                        "--demo": False,
                                                        "--hyperopt": False,
                                                        "--spot": False,
                                                        "--account": "binanceaccount3",
                                                        "--exchange": "binance",
                                                        "--pair": "BTCUSDT",
                                                        "--strategy": "CandleTester",
                                                        "--session": None}}                                              
}
