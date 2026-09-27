//+------------------------------------------------------------------+
//| VSA_ML_Signal_Bridge_EA.mq5                                      |
//| Reads a trained Python ML signal CSV and sends MT5 alerts.        |
//| Demo/education only. No profit guarantee.                         |
//+------------------------------------------------------------------+
#property strict
#include <Trade/Trade.mqh>

input string SignalFileName = "ml_signal.csv";
input bool EnableDesktopAlerts = true;
input bool EnablePushNotifications = true;
input bool EnableDemoTrading = false;
input bool ShowChartEvidence = true;
input int TimerSeconds = 10;
input long MagicNumber = 260817;
input int MaxSignalAgeMinutes = 15;

CTrade trade;
string last_signal_time = "";
string gv_prefix;
string object_prefix = "VSA_ML_EVIDENCE_";

string SignalId(string signal_time, string direction, int scenario_id)
{
   string id = signal_time;
   StringReplace(id, "-", "");
   StringReplace(id, ".", "");
   StringReplace(id, ":", "");
   StringReplace(id, " ", "");
   return id + "-" + direction + "-S" + IntegerToString(scenario_id);
}

void DrawPriceLine(string suffix, double price, color line_color, ENUM_LINE_STYLE style)
{
   string name = object_prefix + suffix;
   if(ObjectFind(0, name) < 0) ObjectCreate(0, name, OBJ_HLINE, 0, 0, price);
   ObjectSetDouble(0, name, OBJPROP_PRICE, price);
   ObjectSetInteger(0, name, OBJPROP_COLOR, line_color);
   ObjectSetInteger(0, name, OBJPROP_STYLE, style);
   ObjectSetInteger(0, name, OBJPROP_WIDTH, 2);
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
}

void DrawSignalEvidence(
   string signal_id, datetime cab_time, datetime signal_time, string model,
   string direction, int scenario_id, double probability, double threshold,
   string advisor_output, double entry, double sl, double tp1, double tp4,
   double relative_spread, double relative_volume, double breakout_strength,
   double risk_percent, string news_status
)
{
   if(!ShowChartEvidence) return;
   DrawPriceLine("ENTRY", entry, clrDodgerBlue, STYLE_SOLID);
   DrawPriceLine("STOP", sl, clrTomato, STYLE_DASH);
   DrawPriceLine("TP1", tp1, clrLimeGreen, STYLE_DOT);
   DrawPriceLine("TP4", tp4, clrGreen, STYLE_DASHDOT);

   string cab_name = object_prefix + "CAB_TIME";
   if(ObjectFind(0, cab_name) < 0) ObjectCreate(0, cab_name, OBJ_VLINE, 0, cab_time, 0);
   ObjectMove(0, cab_name, 0, cab_time, 0);
   ObjectSetInteger(0, cab_name, OBJPROP_COLOR, clrOrange);
   ObjectSetInteger(0, cab_name, OBJPROP_STYLE, STYLE_DOT);

   string signal_name = object_prefix + "SIGNAL_TIME";
   if(ObjectFind(0, signal_name) < 0) ObjectCreate(0, signal_name, OBJ_VLINE, 0, signal_time, 0);
   ObjectMove(0, signal_name, 0, signal_time, 0);
   ObjectSetInteger(0, signal_name, OBJPROP_COLOR, clrAqua);

   string panel = object_prefix + "PANEL";
   if(ObjectFind(0, panel) < 0) ObjectCreate(0, panel, OBJ_LABEL, 0, 0, 0);
   ObjectSetInteger(0, panel, OBJPROP_CORNER, CORNER_LEFT_UPPER);
   ObjectSetInteger(0, panel, OBJPROP_XDISTANCE, 12);
   ObjectSetInteger(0, panel, OBJPROP_YDISTANCE, 24);
   ObjectSetInteger(0, panel, OBJPROP_COLOR, clrWhite);
   ObjectSetInteger(0, panel, OBJPROP_FONTSIZE, 10);
   ObjectSetString(0, panel, OBJPROP_FONT, "Consolas");
   string text = "VSA + TRAINED ML LIVE EVIDENCE\n"
                 "Signal ID: " + signal_id + "\n"
                 "VSA: Scenario " + IntegerToString(scenario_id) + " " + direction + "\n"
                 "Model: " + model + " predict_proba=" + DoubleToString(probability, 3) +
                 " threshold=" + DoubleToString(threshold, 3) + "\n"
                 "Decision: " + advisor_output + " | risk=" + DoubleToString(risk_percent, 1) + "%\n"
                 "Rel spread=" + DoubleToString(relative_spread, 2) +
                 " rel volume=" + DoubleToString(relative_volume, 2) +
                 " breakout=" + DoubleToString(breakout_strength, 2) + "\n"
                 "News: " + news_status + " | Orange=CAB, Aqua=signal";
   ObjectSetString(0, panel, OBJPROP_TEXT, text);
   ChartRedraw();
}

int OnInit()
{
   trade.SetExpertMagicNumber(MagicNumber);
   gv_prefix = "VSA_ML_" + IntegerToString((int)MagicNumber) + "_" + _Symbol + "_";
   EventSetTimer(TimerSeconds);
   Print("VSA ML Signal Bridge EA started. Reading Common Files: ", SignalFileName);
   return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason)
{
   EventKillTimer();
   ObjectsDeleteAll(0, object_prefix);
}

double ToDouble(string s)
{
   StringReplace(s, "%", "");
   return StringToDouble(s);
}

double NormalizeVolume(string symbol, double volume)
{
   double min_volume = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
   double max_volume = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX);
   double step = SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP);
   if(step <= 0) return 0.0;
   volume = MathFloor(volume / step) * step;
   return MathMax(min_volume, MathMin(max_volume, volume));
}

double RiskBasedVolume(string symbol, double entry, double stop, double risk_percent)
{
   double tick_size = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_SIZE);
   double tick_value = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_VALUE);
   double distance = MathAbs(entry - stop);
   if(tick_size <= 0 || tick_value <= 0 || distance <= 0 || risk_percent <= 0) return 0.0;
   double risk_money = AccountInfoDouble(ACCOUNT_BALANCE) * risk_percent / 100.0;
   double loss_per_lot = distance / tick_size * tick_value;
   double raw_volume = risk_money / loss_per_lot;
   if(raw_volume < SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN)) return 0.0;
   return NormalizeVolume(symbol, raw_volume);
}

void ManageOpenPosition()
{
   string symbol = _Symbol;
   if(!PositionSelect(symbol) || !GlobalVariableCheck(gv_prefix + "TP1")) return;
   if((long)PositionGetInteger(POSITION_MAGIC) != MagicNumber) return;
   if(GlobalVariableGet(gv_prefix + "PARTIAL") > 0.5) return;

   double tp1 = GlobalVariableGet(gv_prefix + "TP1");
   double entry = PositionGetDouble(POSITION_PRICE_OPEN);
   double current = PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY
                    ? SymbolInfoDouble(symbol, SYMBOL_BID)
                    : SymbolInfoDouble(symbol, SYMBOL_ASK);
   bool reached = PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY ? current >= tp1 : current <= tp1;
   if(!reached) return;

   double volume = PositionGetDouble(POSITION_VOLUME);
   double min_volume = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
   double close_volume = NormalizeVolume(symbol, volume * 0.70);
   if(close_volume >= volume || volume - close_volume < min_volume)
   {
      Print("Partial close skipped because broker minimum volume cannot preserve a 30% runner.");
      return;
   }
   if(!trade.PositionClosePartial(symbol, close_volume))
   {
      Print("70% partial close failed: ", trade.ResultRetcodeDescription());
      return;
   }
   double runner_tp = GlobalVariableGet(gv_prefix + "TP4");
   if(!trade.PositionModify(symbol, entry, runner_tp))
      Print("Breakeven stop modification failed: ", trade.ResultRetcodeDescription());
   else
   {
      GlobalVariableSet(gv_prefix + "PARTIAL", 1.0);
      Print("Closed 70% at 1R; remaining position stop moved to breakeven and TP remains at 4R.");
   }
}

void OnTimer()
{
   ManageOpenPosition();
   int handle = FileOpen(
      SignalFileName,
      FILE_READ|FILE_CSV|FILE_ANSI|FILE_COMMON,
      ',',
      CP_UTF8
   );
   if(handle == INVALID_HANDLE)
   {
      Print("Could not open signal file in Common Files: ", SignalFileName, " error=", GetLastError());
      return;
   }

   // FILE_CSV returns one field per FileReadString call, not a whole row.
   int header_fields = 0;
   for(int i=0; i<29 && !FileIsEnding(handle); i++)
   {
      FileReadString(handle);
      header_fields++;
   }
   if(header_fields != 29 || FileIsEnding(handle))
   {
      Print("Signal CSV header/data boundary invalid. Header fields read=", header_fields,
            ". Expected 29 fields and one data row.");
      FileClose(handle);
      return;
   }

   string signal_time = FileReadString(handle);
   string cab_time_text = FileReadString(handle);
   string csv_symbol = FileReadString(handle);
   string timeframe = FileReadString(handle);
   string model = FileReadString(handle);
   string direction = FileReadString(handle);
   int scenario_id = (int)StringToInteger(FileReadString(handle));
   double probability = ToDouble(FileReadString(handle));
   double model_threshold = ToDouble(FileReadString(handle));
   string advisor_output = FileReadString(handle);
   double entry = ToDouble(FileReadString(handle));
   double sl = ToDouble(FileReadString(handle));
   double tp1 = ToDouble(FileReadString(handle));
   double relative_spread = ToDouble(FileReadString(handle));
   double relative_volume = ToDouble(FileReadString(handle));
   double close_location = ToDouble(FileReadString(handle));
   double body_ratio = ToDouble(FileReadString(handle));
   bool had_opposite_sweep = StringToInteger(FileReadString(handle)) != 0;
   double breakout_strength = ToDouble(FileReadString(handle));
   double risk_percent = ToDouble(FileReadString(handle));
   double tp4 = ToDouble(FileReadString(handle));
   FileReadString(handle); // raw ML output
   FileReadString(handle); // minutes before news
   FileReadString(handle); // minutes after news
   FileReadString(handle); // event
   FileReadString(handle); // event time
   string news_status = FileReadString(handle);
   string news_blocked_text = FileReadString(handle);
   FileReadString(handle); // news reason
   FileClose(handle);

   if(StringLen(signal_time) < 10 || StringLen(advisor_output) == 0)
   {
      Print("Signal CSV row parse failed. signal_time='", signal_time,
            "' advisor_output='", advisor_output, "'.");
      return;
   }

   string expected_symbol = csv_symbol;
   StringToUpper(expected_symbol);
   string chart_symbol = _Symbol;
   StringToUpper(chart_symbol);
   if(StringFind(chart_symbol, expected_symbol) < 0 || timeframe != "M5" || scenario_id < 1 || scenario_id > 3)
   {
      Print("Signal contract rejected. CSV symbol=", csv_symbol, " chart=", _Symbol,
            " timeframe=", timeframe, " scenario=", scenario_id);
      return;
   }

   string symbol = _Symbol; // supports broker suffixes such as XAUUSD.a
   string mt5_signal_time = signal_time;
   StringReplace(mt5_signal_time, "-", ".");
   datetime parsed_time = StringToTime(mt5_signal_time);
   string mt5_cab_time = cab_time_text;
   StringReplace(mt5_cab_time, "-", ".");
   datetime parsed_cab_time = StringToTime(mt5_cab_time);
   if(parsed_time <= 0)
   {
      Print("Signal time could not be parsed: ", signal_time);
      return;
   }
   if(parsed_time > 0 && (TimeCurrent() - parsed_time) > MaxSignalAgeMinutes * 60)
   {
      Print("Ignoring stale signal from ", signal_time);
      return;
   }

   if(signal_time == last_signal_time) return;
   last_signal_time = signal_time;

   string signal_id = SignalId(signal_time, direction, scenario_id);

   DrawSignalEvidence(
      signal_id, parsed_cab_time, parsed_time, model, direction, scenario_id,
      probability, model_threshold, advisor_output, entry, sl, tp1, tp4,
      relative_spread, relative_volume, breakout_strength, risk_percent, news_status
   );

   string msg = "VSA ML Bot " + model + " | ID=" + signal_id + " | " + symbol + " " + timeframe +
                " | Scenario " + IntegerToString(scenario_id) + " " + direction +
                " | prob=" + DoubleToString(probability, 3) + ">=" + DoubleToString(model_threshold, 3) +
                " | " + advisor_output;

   if(EnableDesktopAlerts) Alert(msg);
   if(EnablePushNotifications) SendNotification(msg);
   Print(msg);

   StringToLower(news_blocked_text);
   bool news_blocked = news_blocked_text == "true" || news_blocked_text == "1";
   if(EnableDemoTrading && advisor_output == "TRADE_CANDIDATE" && probability >= model_threshold && !news_blocked)
   {
      if(PositionSelect(symbol))
      {
         Print("Trade skipped: a position already exists for ", symbol);
         return;
      }
      int digits = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS);
      sl = NormalizeDouble(sl, digits);
      tp1 = NormalizeDouble(tp1, digits);
      tp4 = NormalizeDouble(tp4, digits);
      double market_entry = direction == "BUY" ? SymbolInfoDouble(symbol, SYMBOL_ASK) : SymbolInfoDouble(symbol, SYMBOL_BID);
      double lots = RiskBasedVolume(symbol, market_entry, sl, risk_percent);
      if(lots <= 0)
      {
         Print("Trade skipped: risk-based position size could not be calculated.");
         return;
      }
      bool sent = false;
      if(direction == "BUY") sent = trade.Buy(lots, symbol, 0.0, sl, tp4, "VSA ML BUY");
      if(direction == "SELL") sent = trade.Sell(lots, symbol, 0.0, sl, tp4, "VSA ML SELL");
      if(!sent) Print("Trade request failed: ", trade.ResultRetcode(), " ", trade.ResultRetcodeDescription());
      else
      {
         GlobalVariableSet(gv_prefix + "TP1", tp1);
         GlobalVariableSet(gv_prefix + "TP4", tp4);
         GlobalVariableSet(gv_prefix + "PARTIAL", 0.0);
         Print("Demo trade opened with risk-based volume=", DoubleToString(lots, 2),
               ", risk=", DoubleToString(risk_percent, 1), "%, news=", news_status);
      }
   }
}
