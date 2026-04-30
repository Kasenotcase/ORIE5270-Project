\# Data Documentation



\## Data sources



This project uses real financial and macro-financial data.



\## Asset data



The asset data comes from Yahoo Finance. The default asset universe is a set of Select Sector SPDR ETFs:



\- XLE

\- XLF

\- XLK

\- XLY

\- XLI

\- XLV

\- XLP

\- XLU

\- XLB



For each ETF, the project uses:



\- adjusted close price

\- close price

\- trading volume



Adjusted close prices are used to compute asset returns. Close prices and trading volume are used to compute dollar volume and the Amihud illiquidity feature.



\## Macro-financial data



The project uses two FRED macro-financial indicators:



\- VIXCLS

\- BAMLH0A0HYM2



VIXCLS is used as a market volatility stress proxy.



BAMLH0A0HYM2 is used as a credit-spread stress proxy.



\## Cleaning and alignment



The data-cleaning pipeline does the following:



1\. Removes rows with missing or infinite prices.

2\. Removes rows with invalid adjusted close or close prices.

3\. Allows zero volume but removes negative volume.

4\. Computes daily percentage returns from adjusted close prices.

5\. Forward-fills macro-financial variables to trading dates.

6\. Aligns prices, volume, returns, and macro variables to a common return index.



The relevant function is `regime\_mpc.data.clean\_data`.



\## Testing



The data module is tested using small synthetic and cached datasets. The unit tests avoid relying on live network calls so that tests remain reproducible.

