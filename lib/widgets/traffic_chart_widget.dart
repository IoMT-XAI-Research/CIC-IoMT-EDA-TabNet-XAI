import 'dart:async';
import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';
import '../api_service.dart';
// Assuming main.dart has neonRed, neonGreen etc. We should probably extract valid colors, but for now I'll use hardcoded colors or try to import.
// The user code in main.dart defines constants. I should check main.dart for color definitions or just copy them.

// To avoid circular imports or missing constants, I will redefine basic colors here or use Material ones if main.dart is too coupled.
// However, the user wants "Preserve UI styling".
// I'll assume I can import 'package:iomt_ids/main.dart' or wherever constants are.
// Actually, looking at main.dart, the constants are top-level.
// Let's redefine them locally to be safe and independent.

const Color neonGreen = Color(0xFF39FF14);
const Color neonRed = Color(0xFFFF073A);
const Color cardColor = Color(0xFF1E1E1E);
const Color textMuted = Colors.grey;

class LogVelocityChart extends StatefulWidget {
  const LogVelocityChart({super.key});

  @override
  State<LogVelocityChart> createState() => _LogVelocityChartState();
}

class _LogVelocityChartState extends State<LogVelocityChart> {
  final List<FlSpot> _spots = [];
  final ApiService _api = ApiService();
  Timer? _timer;
  double _xValue = 0;
  bool _isHighTraffic = false;

  @override
  void initState() {
    super.initState();
    // Initialize with 20 zero points
    for (int i = 0; i < 20; i++) {
      _spots.add(FlSpot(i.toDouble(), 0));
    }
    _xValue = 19;
    _startPolling();
  }

  void _startPolling() {
    _timer = Timer.periodic(const Duration(seconds: 2), (timer) async {
      if (!mounted) {
        timer.cancel();
        return;
      }
      try {
        final stats = await _api.getLogTrafficStats();
        final count = (stats['count'] as num).toDouble();
        final status = stats['status'];

        if (mounted) {
          setState(() {
            _xValue++;
            _spots.add(FlSpot(_xValue, count));
            if (_spots.length > 20) {
              _spots.removeAt(0);
            }
            _isHighTraffic = status == 'high';
          });
        }
      } catch (e) {
        debugPrint("Chart Error: $e");
      }
    });
  }

  @override
  void dispose() {
    _timer?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    Color lineColor = _isHighTraffic ? neonRed : neonGreen;

    double maxY = 10;
    if (_spots.isNotEmpty) {
      double currentMax =
          _spots.map((e) => e.y).reduce((a, b) => a > b ? a : b);
      maxY = currentMax + 5;
    }

    return Container(
      height: 200,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: cardColor,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: lineColor.withOpacity(0.5)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text("Log Hızı (Son 5sn)",
              style: TextStyle(color: textMuted, fontSize: 12)),
          Text(_isHighTraffic ? "YÜKSEK ETKİLEŞİM !!" : "Normal Akış",
              style: TextStyle(
                  color: lineColor, fontWeight: FontWeight.bold, fontSize: 16)),
          const SizedBox(height: 10),
          Expanded(
            child: LineChart(
              LineChartData(
                gridData: const FlGridData(show: false),
                titlesData: const FlTitlesData(show: false),
                borderData: FlBorderData(show: false),
                minX: _spots.first.x,
                maxX: _spots.last.x,
                minY: 0,
                maxY: maxY,
                lineBarsData: [
                  LineChartBarData(
                    spots: _spots,
                    isCurved: true,
                    color: lineColor,
                    barWidth: 3,
                    dotData: const FlDotData(show: false),
                    belowBarData: BarAreaData(
                        show: true, color: lineColor.withOpacity(0.1)),
                  ),
                ],
                lineTouchData: LineTouchData(
                  enabled: true,
                  touchTooltipData: LineTouchTooltipData(
                      tooltipBgColor: Colors.black87,
                      getTooltipItems: (List<LineBarSpot> touchedBarSpots) {
                        return touchedBarSpots.map((barSpot) {
                          return LineTooltipItem(
                            '${barSpot.y.toInt()} Logs',
                            const TextStyle(
                                color: Colors.white,
                                fontWeight: FontWeight.bold),
                          );
                        }).toList();
                      }),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
