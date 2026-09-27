//+------------------------------------------------------------------+
//| VSA_M5_Bar_Exporter_EA.mq5                                      |
//| Exports completed XAUUSD M5 candles for Python LR inference.     |
//| Demo/education only. It never opens or modifies trades.          |
//+------------------------------------------------------------------+
#property strict

input string OutputFileName = "xauusd_m5_bars.csv";
input int BarsToExport = 2500;
input int TimerSeconds = 10;

datetime last_exported_closed_bar = 0;

bool IsGoldSymbol()
{
   string symbol = _Symbol;
   StringToUpper(symbol);
   return StringFind(symbol, "XAUUSD") >= 0;
}


void WriteConnectionStatus()
{
   int handle = FileOpen(
      "vsa_mt5_connection_status.csv",
      FILE_WRITE|FILE_CSV|FILE_ANSI|FILE_COMMON,
      ',',
      CP_UTF8
   );
   if(handle == INVALID_HANDLE)
   {
      Print("Could not write MT5 connection status. error=", GetLastError());
      return;
   }
   datetime server_time = TimeCurrent();
   datetime gmt_time = TimeGMT();
   double broker_offset_hours = (double)(server_time - gmt_time) / 3600.0;
   FileWrite(handle, "server_time", "gmt_time", "broker_utc_offset_hours", "symbol", "timeframe", "terminal_connected");
   FileWrite(
      handle,
      TimeToString(server_time, TIME_DATE|TIME_SECONDS),
      TimeToString(gmt_time, TIME_DATE|TIME_SECONDS),
      DoubleToString(broker_offset_hours, 2),
      _Symbol,
      "M5",
      (int)TerminalInfoInteger(TERMINAL_CONNECTED)
   );
   FileFlush(handle);
   FileClose(handle);
}

bool ExportCompletedM5Bars()
{
   if(!IsGoldSymbol())
   {
      Print("VSA M5 exporter must be attached to an XAUUSD chart. Current symbol: ", _Symbol);
      return false;
   }

   // Do not keep presenting cached history as though it were a live export when the
   // terminal has lost its broker connection. The heartbeat is still written so the
   // Python/Streamlit side can show DISCONNECTED explicitly.
   if(!TerminalInfoInteger(TERMINAL_CONNECTED))
   {
      WriteConnectionStatus();
      Print("VSA M5 exporter waiting: MetaTrader 5 is not connected to the broker server.");
      return false;
   }

   datetime newest_closed_bar = iTime(_Symbol, PERIOD_M5, 1);
   if(newest_closed_bar <= 0)
   {
      Print("VSA M5 exporter is waiting for completed M5 history.");
      return false;
   }
   if(newest_closed_bar == last_exported_closed_bar)
   {
      WriteConnectionStatus();
      return true;
   }

   MqlRates rates[];
   int requested = MathMax(BarsToExport, 250);
   int copied = CopyRates(_Symbol, PERIOD_M5, 1, requested, rates);
   if(copied < 250)
   {
      Print("VSA M5 exporter received only ", copied, " completed bars; at least 250 are required.");
      return false;
   }

   int handle = FileOpen(
      OutputFileName,
      FILE_WRITE|FILE_CSV|FILE_ANSI|FILE_COMMON,
      ',',
      CP_UTF8
   );
   if(handle == INVALID_HANDLE)
   {
      Print("Could not create Common Files export: ", OutputFileName, " error=", GetLastError());
      return false;
   }

   FileWrite(handle, "time", "open", "high", "low", "close", "tick_volume", "real_volume", "broker_spread_points");
   for(int i=0; i<copied; i++)
   {
      FileWrite(
         handle,
         TimeToString(rates[i].time, TIME_DATE|TIME_SECONDS),
         DoubleToString(rates[i].open, _Digits),
         DoubleToString(rates[i].high, _Digits),
         DoubleToString(rates[i].low, _Digits),
         DoubleToString(rates[i].close, _Digits),
         (long)rates[i].tick_volume,
         (long)rates[i].real_volume,
         (int)rates[i].spread
      );
   }
   FileFlush(handle);
   FileClose(handle);
   last_exported_closed_bar = newest_closed_bar;
   WriteConnectionStatus();
   Print("VSA M5 exporter wrote ", copied, " completed bars to Common Files: ", OutputFileName,
         ". Latest closed bar: ", TimeToString(newest_closed_bar, TIME_DATE|TIME_SECONDS));
   return true;
}

int OnInit()
{
   if(BarsToExport < 250 || TimerSeconds < 5)
   {
      Print("Invalid exporter settings: BarsToExport must be >=250 and TimerSeconds >=5.");
      return INIT_PARAMETERS_INCORRECT;
   }
   EventSetTimer(TimerSeconds);
   ExportCompletedM5Bars();
   return INIT_SUCCEEDED;
}

void OnTimer()
{
   ExportCompletedM5Bars();
}

void OnDeinit(const int reason)
{
   EventKillTimer();
}
