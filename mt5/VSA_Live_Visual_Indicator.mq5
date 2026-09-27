//+------------------------------------------------------------------+
//| VSA_Live_Visual_Indicator.mq5                                    |
//| Visible MQL5 mirror of the deployed Python CAB/scenario rules.    |
//| Rule evidence only: trained LR probability is shown by the bridge.|
//+------------------------------------------------------------------+
#property strict
#property indicator_chart_window
#property indicator_plots 0

input int FeatureLookback = 20;
input int VolumeLookback = 30;
input double CABVolumeRatio = 1.50;
input int TrendEMAPeriod = 50;
input double MomentumBodyRatio = 0.60;
input double MomentumRangeRatio = 1.20;
input int MaximumBarsAfterCAB = 36;
input int BarsToAudit = 2500;
input bool ExportLatestParityAudit = true;
input string ParityAuditFileName = "vsa_mql5_latest_audit.csv";

int ema_handle = INVALID_HANDLE;
string prefix = "VSA_RULE_EVIDENCE_";
datetime last_rendered_signal = 0;

double PreviousAverageRange(const MqlRates &rates[], int shift, int lookback, int copied)
{
   if(shift + lookback >= copied) return 0.0;
   double total = 0.0;
   for(int j=1; j<=lookback; j++) total += rates[shift+j].high - rates[shift+j].low;
   return total / lookback;
}

double PreviousAverageVolume(const MqlRates &rates[], int shift, int lookback, int copied)
{
   if(shift + lookback >= copied) return 0.0;
   double total = 0.0;
   for(int j=1; j<=lookback; j++) total += (double)rates[shift+j].tick_volume;
   return total / lookback;
}

void TrendLine(string suffix, datetime t1, datetime t2, double price, color line_color)
{
   string name = prefix + suffix;
   if(ObjectFind(0, name) < 0) ObjectCreate(0, name, OBJ_TREND, 0, t1, price, t2, price);
   ObjectMove(0, name, 0, t1, price);
   ObjectMove(0, name, 1, t2, price);
   ObjectSetInteger(0, name, OBJPROP_RAY_RIGHT, true);
   ObjectSetInteger(0, name, OBJPROP_COLOR, line_color);
   ObjectSetInteger(0, name, OBJPROP_WIDTH, 2);
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
}

void PriceLine(string suffix, double price, color line_color, ENUM_LINE_STYLE style)
{
   string name = prefix + suffix;
   if(ObjectFind(0, name) < 0) ObjectCreate(0, name, OBJ_HLINE, 0, 0, price);
   ObjectSetDouble(0, name, OBJPROP_PRICE, price);
   ObjectSetInteger(0, name, OBJPROP_COLOR, line_color);
   ObjectSetInteger(0, name, OBJPROP_STYLE, style);
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
}

void RenderVSAEvidence(
   datetime cab_time, datetime signal_time, double cab_high, double cab_low,
   string direction, int scenario, double entry, double stop, double tp1, double tp4,
   double cab_volume_ratio, double relative_spread, double body_ratio
)
{
   if(signal_time == last_rendered_signal) return;
   last_rendered_signal = signal_time;
   ObjectsDeleteAll(0, prefix);
   TrendLine("CAB_HIGH", cab_time, signal_time, cab_high, clrOrange);
   TrendLine("CAB_LOW", cab_time, signal_time, cab_low, clrOrange);
   PriceLine("ENTRY", entry, clrDodgerBlue, STYLE_SOLID);
   PriceLine("STOP", stop, clrTomato, STYLE_DASH);
   PriceLine("TP1", tp1, clrLimeGreen, STYLE_DOT);
   PriceLine("TP4", tp4, clrGreen, STYLE_DASHDOT);

   string arrow = prefix + "CONFIRMATION";
   ObjectCreate(0, arrow, direction == "BUY" ? OBJ_ARROW_BUY : OBJ_ARROW_SELL, 0, signal_time, entry);
   ObjectSetInteger(0, arrow, OBJPROP_COLOR, direction == "BUY" ? clrLime : clrTomato);

   string label = prefix + "PANEL";
   ObjectCreate(0, label, OBJ_LABEL, 0, 0, 0);
   ObjectSetInteger(0, label, OBJPROP_CORNER, CORNER_RIGHT_UPPER);
   ObjectSetInteger(0, label, OBJPROP_XDISTANCE, 12);
   ObjectSetInteger(0, label, OBJPROP_YDISTANCE, 24);
   ObjectSetInteger(0, label, OBJPROP_COLOR, clrWhite);
   ObjectSetInteger(0, label, OBJPROP_FONTSIZE, 10);
   ObjectSetString(0, label, OBJPROP_FONT, "Consolas");
   ObjectSetString(0, label, OBJPROP_TEXT,
      "MQL5 VSA RULE MIRROR (completed M5 bars)\n"
      "Scenario " + IntegerToString(scenario) + " " + direction + "\n"
      "CAB volume ratio=" + DoubleToString(cab_volume_ratio, 2) +
      " | breakout rel spread=" + DoubleToString(relative_spread, 2) +
      " | body=" + DoubleToString(body_ratio, 2) + "\n"
      "Orange=CAB zone | Blue=entry | Red=stop | Green=targets\n"
      "ML probability is NOT calculated here; Python bridge supplies trained LR output."
   );
   if(ExportLatestParityAudit)
   {
      int handle = FileOpen(ParityAuditFileName, FILE_WRITE|FILE_CSV|FILE_ANSI|FILE_COMMON, ',', CP_UTF8);
      if(handle != INVALID_HANDLE)
      {
         FileWrite(handle, "signal_time", "cab_time", "direction", "scenario_id", "cab_high", "cab_low",
                   "entry_price", "stop_price", "target_price", "runner_target_price",
                   "cab_volume_ratio", "relative_spread", "body_ratio");
         FileWrite(handle, TimeToString(signal_time, TIME_DATE|TIME_SECONDS),
                   TimeToString(cab_time, TIME_DATE|TIME_SECONDS), direction, scenario,
                   cab_high, cab_low, entry, stop, tp1, tp4,
                   cab_volume_ratio, relative_spread, body_ratio);
         FileFlush(handle);
         FileClose(handle);
      }
      else Print("Could not write MQL5 VSA parity audit: ", GetLastError());
   }
   ChartRedraw();
}

void ScanCompletedBars()
{
   if(_Period != PERIOD_M5) return;
   MqlRates rates[];
   double ema[];
   ArraySetAsSeries(rates, true);
   ArraySetAsSeries(ema, true);
   int requested = MathMax(BarsToAudit, TrendEMAPeriod + VolumeLookback + 20);
   int copied = CopyRates(_Symbol, PERIOD_M5, 0, requested, rates);
   if(copied < 250) return;
   if(CopyBuffer(ema_handle, 0, 0, copied, ema) != copied) return;

   bool active = false;
   string active_direction = "";
   int cab_shift = -1;
   datetime cab_time = 0;
   double cab_high = 0.0, cab_low = 0.0, cab_volume = 0.0, cab_ratio = 0.0;
   bool first_break = false, reentered = false, swept = false, broken = false;
   double sweep_stop = 0.0, post_break_extreme = 0.0;

   int oldest = copied - MathMax(VolumeLookback, MathMax(FeatureLookback, TrendEMAPeriod)) - 6;
   for(int shift=oldest; shift>=1; shift--)
   {
      double range = rates[shift].high - rates[shift].low;
      if(range <= 0.0) continue;
      double avg_volume30 = PreviousAverageVolume(rates, shift, VolumeLookback, copied);
      double avg_range20 = PreviousAverageRange(rates, shift, FeatureLookback, copied);
      if(avg_volume30 <= 0.0 || avg_range20 <= 0.0 || shift + 5 >= copied) continue;
      double volume_ratio30 = (double)rates[shift].tick_volume / avg_volume30;
      double relative_spread = range / avg_range20;
      double body_ratio = MathAbs(rates[shift].close-rates[shift].open) / range;
      bool downtrend = rates[shift].close < ema[shift] && ema[shift] - ema[shift+5] < 0.0;
      bool uptrend = rates[shift].close > ema[shift] && ema[shift] - ema[shift+5] > 0.0;
      bool buy_cab = rates[shift].close < rates[shift].open && downtrend && volume_ratio30 >= CABVolumeRatio;
      bool sell_cab = rates[shift].close > rates[shift].open && uptrend && volume_ratio30 >= CABVolumeRatio;

      if(!active)
      {
         if(buy_cab || sell_cab)
         {
            active = true;
            active_direction = buy_cab ? "BUY" : "SELL";
            cab_shift = shift;
            cab_time = rates[shift].time;
            cab_high = rates[shift].high;
            cab_low = rates[shift].low;
            cab_volume = (double)rates[shift].tick_volume;
            cab_ratio = volume_ratio30;
            first_break = false; reentered = false; swept = false; broken = false;
            sweep_stop = 0.0; post_break_extreme = 0.0;
         }
         continue;
      }

      int bars_after = cab_shift - shift;
      if(bars_after > MaximumBarsAfterCAB)
      {
         active = false;
         continue;
      }

      bool lower_volume = (double)rates[shift].tick_volume < cab_volume;
      bool bull_momentum = rates[shift].close > rates[shift].open &&
                           body_ratio >= MomentumBodyRatio && relative_spread >= MomentumRangeRatio;
      bool bear_momentum = rates[shift].close < rates[shift].open &&
                           body_ratio >= MomentumBodyRatio && relative_spread >= MomentumRangeRatio;
      bool valid_breakout = false, inside = false;

      if(active_direction == "BUY")
      {
         if(broken) post_break_extreme = MathMin(post_break_extreme, rates[shift].low);
         else if(rates[shift].close < cab_low)
         {
            broken = true;
            post_break_extreme = rates[shift].low;
         }
         else if(rates[shift].low < cab_low && rates[shift].close >= cab_low)
         {
            swept = true;
            sweep_stop = rates[shift].low;
         }
         valid_breakout = rates[shift].close > cab_high && bull_momentum && lower_volume;
         inside = rates[shift].close <= cab_high && rates[shift].close >= cab_low;
      }
      else
      {
         if(broken) post_break_extreme = MathMax(post_break_extreme, rates[shift].high);
         else if(rates[shift].close > cab_high)
         {
            broken = true;
            post_break_extreme = rates[shift].high;
         }
         else if(rates[shift].high > cab_high && rates[shift].close <= cab_high)
         {
            swept = true;
            sweep_stop = rates[shift].high;
         }
         valid_breakout = rates[shift].close < cab_low && bear_momentum && lower_volume;
         inside = rates[shift].close >= cab_low && rates[shift].close <= cab_high;
      }

      int scenario = 0;
      if(valid_breakout && broken) scenario = 3;
      else if(valid_breakout && swept) scenario = 2;
      else if(valid_breakout && first_break && reentered) scenario = 1;
      else if(valid_breakout)
      {
         first_break = true;
         continue;
      }
      else if(first_break && inside && !swept && !broken) reentered = true;
      if(scenario == 0) continue;

      double entry = rates[shift].close;
      double stop = scenario == 1 ? (active_direction == "BUY" ? cab_low : cab_high)
                    : scenario == 2 ? sweep_stop : post_break_extreme;
      double risk = MathMax(MathAbs(entry-stop), 0.50);
      double tp1 = active_direction == "BUY" ? entry+risk : entry-risk;
      double tp4 = active_direction == "BUY" ? entry+4.0*risk : entry-4.0*risk;
      RenderVSAEvidence(cab_time, rates[shift].time, cab_high, cab_low, active_direction,
                        scenario, entry, stop, tp1, tp4, cab_ratio, relative_spread, body_ratio);
      active = false;
   }
}

int OnInit()
{
   if(_Period != PERIOD_M5)
   {
      Print("VSA visual indicator requires an M5 chart.");
      return INIT_PARAMETERS_INCORRECT;
   }
   ema_handle = iMA(_Symbol, PERIOD_M5, TrendEMAPeriod, 0, MODE_EMA, PRICE_CLOSE);
   if(ema_handle == INVALID_HANDLE) return INIT_FAILED;
   IndicatorSetString(INDICATOR_SHORTNAME, "VSA Live Rule Mirror");
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
   if(ema_handle != INVALID_HANDLE) IndicatorRelease(ema_handle);
   ObjectsDeleteAll(0, prefix);
}

int OnCalculate(
   const int rates_total, const int prev_calculated, const datetime &time[],
   const double &open[], const double &high[], const double &low[], const double &close[],
   const long &tick_volume[], const long &volume[], const int &spread[]
)
{
   ScanCompletedBars();
   return rates_total;
}
