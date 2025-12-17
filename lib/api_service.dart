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
        final prefs = await SharedPreferences.getInstance();
        await prefs.setString('access_token', token);
        print('Token saved successfully');
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

    if (response.statusCode != 201) {
      throw Exception('Failed to create hospital: ${response.body}');
    }
  }

  Future<void> createDevice(String name, String ip, String hospitalCode) async {
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
        'hospital_unique_code': hospitalCode,
        'status': 'SAFE'
      }),
    );

    if (response.statusCode != 201) {
      throw Exception('Failed to create device: ${response.body}');
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
