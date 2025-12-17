import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';

class ApiService {
  // Using Render Production URL
  static const String baseUrl = 'https://cic-iomt-eda-tabnet-xai.onrender.com';

  Future<void> login(String email, String password) async {
    final url = Uri.parse('$baseUrl/auth/login');
    print('Attempting login to: $url');
    print('Email: $email');

    try {
      // OAuth2PasswordRequestForm expects x-www-form-urlencoded
      final response = await http.post(
        url,
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: {
          'username': email, // Mapping email to username field
          'password': password,
        },
      );

      print('Login Response Status: ${response.statusCode}');
      print('Login Response Body: ${response.body}');

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        final token = data['access_token'];
        final role = data['role'] ?? 'TECH_STAFF'; // Default to tech

        final prefs = await SharedPreferences.getInstance();
        await prefs.setString('access_token', token);
        await prefs.setString('user_role', role);
        print('Token and Role ($role) saved successfully');
      } else {
        throw Exception('Failed to login: ${response.body}');
      }
    } catch (e) {
      print('Login Error: $e');
      rethrow;
    }
  }

  Future<List<dynamic>> getDevices(String hospitalUniqueCode) async {
    final prefs = await SharedPreferences.getInstance();
    final token = prefs.getString('access_token');
    if (token == null) throw Exception('No token found');

    final url =
        Uri.parse('$baseUrl/devices/?hospital_unique_code=$hospitalUniqueCode');
    final response = await http.get(
      url,
      headers: {
        'Authorization': 'Bearer $token',
      },
    );

    if (response.statusCode == 200) {
      return jsonDecode(response.body);
    } else {
      throw Exception('Failed to load devices: ${response.body}');
    }
  }

  Future<List<dynamic>> getHospitals() async {
    final prefs = await SharedPreferences.getInstance();
    final token = prefs.getString('access_token');
    if (token == null) throw Exception('No token found');

    final url = Uri.parse('$baseUrl/hospitals/');
    final response = await http.get(
      url,
      headers: {'Authorization': 'Bearer $token'},
    );

    if (response.statusCode == 200) {
      return jsonDecode(response.body);
    } else {
      throw Exception('Failed to load hospitals: ${response.body}');
    }
  }

  Future<void> createHospital(String name, String uniqueCode) async {
    final prefs = await SharedPreferences.getInstance();
    final token = prefs.getString('access_token');
    if (token == null) throw Exception('No token found');

    final url = Uri.parse('$baseUrl/hospitals/');
    final response = await http.post(
      url,
      headers: {
        'Authorization': 'Bearer $token',
        'Content-Type': 'application/json',
      },
      body: jsonEncode({'name': name, 'unique_code': uniqueCode}),
    );

    if (response.statusCode != 200 && response.statusCode != 201) {
      throw Exception('Failed to create hospital: ${response.body}');
    }
  }

  Future<void> createDevice(
      String name, String ip, String? room, String hospitalCode) async {
    final prefs = await SharedPreferences.getInstance();
    final token = prefs.getString('access_token');
    if (token == null) throw Exception('No token found');

    final url = Uri.parse('$baseUrl/devices/');
    final response = await http.post(
      url,
      headers: {
        'Authorization': 'Bearer $token',
        'Content-Type': 'application/json',
      },
      body: jsonEncode({
        'name': name,
        'ip_address': ip,
        'room_number': room,
        'hospital_unique_code': hospitalCode,
        'status': 'SAFE'
      }),
    );

    if (response.statusCode != 200 && response.statusCode != 201) {
      throw Exception('Failed to create device: ${response.body}');
    }
  }

  Future<void> updateHospital(int id, String name, String uniqueCode) async {
    final prefs = await SharedPreferences.getInstance();
    final token = prefs.getString('access_token');
    if (token == null) throw Exception('No token found');

    final url = Uri.parse('$baseUrl/hospitals/$id');
    final response = await http.put(
      url,
      headers: {
        'Authorization': 'Bearer $token',
        'Content-Type': 'application/json',
      },
      body: jsonEncode({'name': name, 'unique_code': uniqueCode}),
    );

    if (response.statusCode != 200) {
      throw Exception('Failed to update hospital: ${response.body}');
    }
  }

  Future<void> deleteHospital(int id) async {
    final prefs = await SharedPreferences.getInstance();
    final token = prefs.getString('access_token');
    if (token == null) throw Exception('No token found');

    final url = Uri.parse('$baseUrl/hospitals/$id');
    final response = await http.delete(
      url,
      headers: {
        'Authorization': 'Bearer $token',
      },
    );

    if (response.statusCode != 204) {
      throw Exception('Failed to delete hospital: ${response.body}');
    }
  }

  Future<void> deleteDevice(int id) async {
    final prefs = await SharedPreferences.getInstance();
    final token = prefs.getString('access_token');
    if (token == null) throw Exception('No token found');

    final url = Uri.parse('$baseUrl/devices/$id');
    final response = await http.delete(
      url,
      headers: {
        'Authorization': 'Bearer $token',
      },
    );

    if (response.statusCode != 204) {
      throw Exception('Failed to delete device: ${response.body}');
    }
  }

  Future<List<dynamic>> getLogs() async {
    final prefs = await SharedPreferences.getInstance();
    final token = prefs.getString('access_token');
    if (token == null) throw Exception('No token found');

    final url = Uri.parse('$baseUrl/logs/');
    final response = await http.get(
      url,
      headers: {'Authorization': 'Bearer $token'},
    );

    if (response.statusCode == 200) {
      return jsonDecode(response.body);
    } else {
      throw Exception('Failed to load logs: ${response.body}');
    }
  }

  Future<Map<String, dynamic>> getStats() async {
    // Basic implementation: Count hospitals and devices
    // Note: Ideally backend should provide a stats endpoint
    try {
      final hospitals = await getHospitals();
      int deviceCount = 0;

      // Fetch devices for each hospital to get accurate count
      // This is inefficient but works with current backend
      for (var h in hospitals) {
        if (h['unique_code'] != null) {
          try {
            final devices = await getDevices(h['unique_code']);
            deviceCount += devices.length;
          } catch (e) {
            print('Error fetching devices for ${h['name']}: $e');
          }
        }
      }

      return {
        'hospital_count': hospitals.length,
        'device_count': deviceCount,
        'alert_count': 0 // Placeholder
      };
    } catch (e) {
      print('getStats error: $e');
      return {'hospital_count': 0, 'device_count': 0, 'alert_count': 0};
    }
  }

  Future<Map<String, dynamic>> getAnalysis(int deviceId) async {
    final prefs = await SharedPreferences.getInstance();
    final token = prefs.getString('access_token');
    if (token == null) throw Exception('No token found');

    final url = Uri.parse('$baseUrl/devices/$deviceId/analysis/latest');
    final response = await http.get(
      url,
      headers: {
        'Authorization': 'Bearer $token',
      },
    );

    if (response.statusCode == 200) {
      return jsonDecode(response.body);
    } else {
      throw Exception('Failed to load analysis: ${response.body}');
    }
  }

  Future<void> isolateDevice(int deviceId) async {
    final prefs = await SharedPreferences.getInstance();
    final token = prefs.getString('access_token');
    if (token == null) throw Exception('No token found');

    final url = Uri.parse('$baseUrl/devices/$deviceId/isolate');
    final response = await http.post(
      url,
      headers: {
        'Authorization': 'Bearer $token',
      },
    );

    if (response.statusCode != 200) {
      throw Exception('Failed to isolate device: ${response.body}');
    }
  }
}
