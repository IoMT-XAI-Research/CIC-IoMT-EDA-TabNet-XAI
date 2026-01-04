import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import '../api_service.dart';

// Assuming basic colors are standard or passed.
// For consistency, reapplying local constants matching the dark theme.
const Color neonGreen = Color(0xFF39FF14);
const Color neonRed = Color(0xFFFF073A);
const Color neonBlue = Color(0xFF00F0FF);
const Color neonOrange = Color(0xFFFFA500);
const Color cardColor = Color(0xFF1E1E1E);
const Color textLight = Colors.white;
const Color textMuted = Colors.grey;

class RecentEventsList extends StatefulWidget {
  const RecentEventsList({super.key});

  @override
  State<RecentEventsList> createState() => _RecentEventsListState();
}

class _RecentEventsListState extends State<RecentEventsList> {
  final ApiService _api = ApiService();
  List<dynamic> _logs = [];
  bool _isLoading = true;

  @override
  void initState() {
    super.initState();
    _loadLogs();
  }

  Future<void> _loadLogs() async {
    try {
      final logs = await _api.fetchRecentLogs(limit: 3);
      if (mounted) {
        setState(() {
          _logs = logs;
          _isLoading = false;
        });
      }
    } catch (e) {
      debugPrint("Error loading recent logs: $e");
      if (mounted) {
        setState(() => _isLoading = false);
      }
    }
  }

  IconData _getIconForType(String type) {
    if (type.contains('SUCCESS')) return Icons.check_circle_outline;
    if (type.contains('WARNING')) return Icons.warning_amber_rounded;
    if (type.contains('DANGER') || type.contains('ATTACK'))
      return Icons.dangerous;
    return Icons.info_outline;
  }

  Color _getColorForType(String type) {
    if (type.contains('SUCCESS')) return neonGreen;
    if (type.contains('WARNING')) return neonOrange;
    if (type.contains('DANGER') || type.contains('ATTACK')) return neonRed;
    return neonBlue;
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: cardColor,
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            "Son Olaylar",
            style: TextStyle(
              fontSize: 18,
              fontWeight: FontWeight.bold,
              color: textLight,
            ),
          ),
          const SizedBox(height: 10),
          if (_isLoading)
            const Center(child: CircularProgressIndicator(color: neonBlue))
          else if (_logs.isEmpty)
            const Padding(
              padding: EdgeInsets.all(20.0),
              child: Center(
                  child: Text("Henüz kayıt yok.",
                      style: TextStyle(color: textMuted))),
            )
          else
            ListView.separated(
              shrinkWrap: true,
              physics: const NeverScrollableScrollPhysics(),
              itemCount: _logs.length,
              separatorBuilder: (context, index) =>
                  const Divider(color: Colors.white10),
              itemBuilder: (context, index) {
                final log = _logs[index];
                final String logType = log['log_type'] ?? 'INFO';
                final String title = log['title'] ?? 'Olay';
                final String rawTime = log['timestamp'] ?? '';

                // Format Time (HH:mm)
                String displayTime = rawTime;
                try {
                  final dt = DateTime.parse(rawTime);
                  displayTime = DateFormat('HH:mm').format(dt.toLocal());
                } catch (_) {}

                return ListTile(
                  leading: Icon(
                    _getIconForType(logType),
                    color: _getColorForType(logType),
                    size: 28,
                  ),
                  title: Text(
                    title,
                    style: const TextStyle(
                        color: textLight, fontWeight: FontWeight.bold),
                  ),
                  subtitle: Text(
                    log['description'] ?? '',
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                    style: const TextStyle(color: textMuted, fontSize: 12),
                  ),
                  trailing: Text(
                    displayTime,
                    style: const TextStyle(color: textMuted, fontSize: 12),
                  ),
                  contentPadding: EdgeInsets.zero,
                  dense: true,
                );
              },
            ),
        ],
      ),
    );
  }
}
