import 'package:flutter/material.dart';

class EditProfile extends StatefulWidget {
  const EditProfile({super.key});

  @override
  State<EditProfile> createState() => _EditProfileState();
}

class _EditProfileState extends State<EditProfile> {
  final TextEditingController _nameController = TextEditingController();
  final TextEditingController _emailController = TextEditingController();
  final TextEditingController _mpdController = TextEditingController();
  final TextEditingController _dpwController = TextEditingController();
  final TextEditingController _heightController = TextEditingController();
  final TextEditingController _weightController = TextEditingController();

  DateTime birthDate = DateTime.utc(2019, 01, 01);
  TimeOfDay gib = TimeOfDay(hour: 22, minute: 30);
  TimeOfDay wu = TimeOfDay(hour: 06, minute: 30);

  var selectGen = 0;

  @override
  Widget build(BuildContext context) {
    final size = MediaQuery.of(context).size;
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.white,
        elevation: 0,
        title: Text("Edit Profile"),
        centerTitle: true,
        leading: IconButton(
          onPressed: () {
                              Navigator.pop(context);

          },
          icon: Icon(Icons.arrow_back, size: 30, weight: 3),
        ),
      ),
      body: SingleChildScrollView(
        child: Container(
          width: double.infinity,
          height: size.height * 1.1,
          color: Colors.white,
          padding: EdgeInsets.symmetric(horizontal: 18),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisAlignment: MainAxisAlignment.spaceEvenly,
            children: [
              Text("About Me",
                  style: TextStyle(
                      color: Colors.black,
                      fontWeight: FontWeight.w600,
                      fontSize: 22)),
              Text("Name",
                  style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600)),
              Container(
                decoration: BoxDecoration(
                  color: Color.fromRGBO(74, 213, 205, 0.1),
                ),
                margin: EdgeInsets.symmetric(vertical: 10),
                padding: EdgeInsets.symmetric(vertical: 7, horizontal: 25),
                child: TextFormField(
                  controller: _nameController,
                  keyboardType: TextInputType.name,
                  decoration: const InputDecoration(
                    border: InputBorder.none,
                    hintText: "Enter Name",
                  ),
                ),
              ),
              SizedBox(height: 2),
              Text("Email",
                  style:
                      TextStyle(fontSize: 16, fontWeight: FontWeight.w600)),
              Container(
                decoration: BoxDecoration(
                  color: Color.fromRGBO(74, 213, 205, 0.1),
                ),
                margin: EdgeInsets.symmetric(vertical: 10),
                padding: EdgeInsets.symmetric(vertical: 7, horizontal: 25),
                child: TextFormField(
                  enabled: false,
                  controller: _emailController,
                  keyboardType: TextInputType.emailAddress,
                  decoration: const InputDecoration(
                      border: InputBorder.none, hintText: "Enter Email"),
                ),
              ),
              SizedBox(height: 2),
              Text("Gender",
                  style:
                      TextStyle(fontSize: 16, fontWeight: FontWeight.w600)),
              GenderSelect(),
              SizedBox(height: 2),
              Text("Date of Birth",
                  style:
                      TextStyle(fontSize: 16, fontWeight: FontWeight.w600)),
              InkWell(
                onTap: () {},
                child: Container(
                    height: 60,
                    width: double.infinity,
                    decoration: BoxDecoration(
                      color: Color.fromRGBO(74, 213, 205, 0.1),
                    ),
                    margin: EdgeInsets.symmetric(vertical: 10),
                    padding: EdgeInsets.symmetric(vertical: 7, horizontal: 25),
                    child: Center(
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          Text(
                            "DD/MM/YYYY",
                            style: TextStyle(
                                color: Colors.black, fontSize: 16),
                          ),
                          Icon(Icons.calendar_month,
                              color: Colors.grey, size: 24)
                        ],
                      ),
                    )),
              ),
              SizedBox(height: 5),
              Text("Other",
                  style: TextStyle(
                      color: Colors.black,
                      fontWeight: FontWeight.w600,
                      fontSize: 22)),
              OtherEdit(size),
              SizedBox(height: 5),
              InkWell(
                onTap: () {},
                child: Container(
                  width: double.infinity,
                  height: size.height * 0.07,
                  decoration: BoxDecoration(
                      color: Color.fromRGBO(74, 213, 205, 1),
                      borderRadius: BorderRadius.circular(100)),
                  margin: EdgeInsets.symmetric(vertical: 10),
                  padding: EdgeInsets.symmetric(vertical: 5, horizontal: 25),
                  child: Center(
                    child: Text("Save",
                        style: TextStyle(
                            color: Colors.white,
                            fontWeight: FontWeight.w700,
                            fontSize: 22)),
                  ),
                ),
              )
            ],
          ),
        ),
      ),
    );
  }

  Widget GenderSelect() {
    return Row(
      children: [
        Expanded(
          child: InkWell(
            onTap: () {},
            child: Container(
              height: 60,
              margin: EdgeInsets.all(5),
              color: Color.fromRGBO(74, 213, 205, 0.1),
              child: Center(
                child: Text("Male",
                    style: TextStyle(color: Colors.black, fontSize: 16)),
              ),
            ),
          ),
        ),
        Expanded(
          child: InkWell(
            onTap: () {},
            child: Container(
              height: 60,
              margin: EdgeInsets.all(5),
              color: Color.fromRGBO(74, 213, 205, 0.1),
              child: Center(
                child: Text("Female",
                    style: TextStyle(color: Colors.black, fontSize: 16)),
              ),
            ),
          ),
        ),
        Expanded(
          child: InkWell(
            onTap: () {},
            child: Container(
              height: 60,
              margin: EdgeInsets.all(5),
              color: Color.fromRGBO(74, 213, 205, 0.1),
              child: Center(
                child: Text("Other",
                    style: TextStyle(color: Colors.black, fontSize: 16)),
              ),
            ),
          ),
        ),
      ],
    );
  }

  Widget OtherEdit(var size) {
    return Container(
      width: double.infinity,
      height: size.height * 0.24,
      child: GridView(
        gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
            crossAxisCount: 2,
            crossAxisSpacing: 2,
            mainAxisSpacing: 2,
            mainAxisExtent: size.height * 0.12),
        children: [
          Container(
            padding: EdgeInsets.symmetric(horizontal: 10, vertical: 10),
            height: size.height * 0.12,
            decoration: BoxDecoration(
                color: Color.fromRGBO(74, 213, 205, 0.1)),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text("Weight(kg)",
                    style: TextStyle(
                        color: Colors.grey,
                        fontSize: 14,
                        fontWeight: FontWeight.w600)),
                TextField(
                  controller: _weightController,
                  style: TextStyle(color: Colors.black, fontSize: 18),
                  keyboardType: TextInputType.number,
                  decoration: const InputDecoration(
                      border: InputBorder.none, hintText: "00"),
                )
              ],
            ),
          ),
          Container(
            padding: EdgeInsets.symmetric(horizontal: 10, vertical: 10),
            height: size.height * 0.12,
            decoration: BoxDecoration(
                color: Color.fromRGBO(74, 213, 205, 0.1)),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text("Height(cm)",
                    style: TextStyle(
                        color: Colors.grey,
                        fontSize: 14,
                        fontWeight: FontWeight.w600)),
                TextField(
                  controller: _heightController,
                  style: TextStyle(color: Colors.black, fontSize: 18),
                  keyboardType: TextInputType.number,
                  decoration: const InputDecoration(
                      border: InputBorder.none, hintText: "00"),
                )
              ],
            ),
          ),
          Container(
            padding: EdgeInsets.symmetric(horizontal: 10, vertical: 10),
            height: size.height * 0.12,
            decoration: BoxDecoration(
                color: Color.fromRGBO(74, 213, 205, 0.1)),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text("Medicine per day",
                    style: TextStyle(
                        color: Colors.grey,
                        fontSize: 14,
                        fontWeight: FontWeight.w600)),
                TextField(
                  controller: _mpdController,
                  style: TextStyle(color: Colors.black, fontSize: 18),
                  keyboardType: TextInputType.number,
                  decoration: const InputDecoration(
                      border: InputBorder.none, hintText: "00"),
                )
              ],
            ),
          ),
          Container(
            padding: EdgeInsets.symmetric(horizontal: 10, vertical: 10),
            height: size.height * 0.12,
            decoration: BoxDecoration(
                color: Color.fromRGBO(74, 213, 205, 0.1)),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text("Diagnosis per week",
                    style: TextStyle(
                        color: Colors.grey,
                        fontSize: 14,
                        fontWeight: FontWeight.w600)),
                TextField(
                  controller: _dpwController,
                  style: TextStyle(color: Colors.black, fontSize: 18),
                  keyboardType: TextInputType.number,
                  decoration: const InputDecoration(
                      border: InputBorder.none, hintText: "00"),
                )
              ],
            ),
          ),
          InkWell(
            onTap: () {},
            child: Container(
              padding: EdgeInsets.symmetric(horizontal: 10, vertical: 10),
              height: size.height * 0.12,
              decoration: BoxDecoration(
                  color: Color.fromRGBO(74, 213, 205, 0.1)),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text("Get in bed",
                      style: TextStyle(
                          color: Colors.grey,
                          fontSize: 14,
                          fontWeight: FontWeight.w600)),
                  Text("22:30",
                      style: TextStyle(color: Colors.black, fontSize: 18)),
                ],
              ),
            ),
          ),
          InkWell(
            onTap: () {},
            child: Container(
              padding: EdgeInsets.symmetric(horizontal: 10, vertical: 10),
              height: size.height * 0.12,
              decoration: BoxDecoration(
                  color: Color.fromRGBO(74, 213, 205, 0.1)),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text("Wake up",
                      style: TextStyle(
                          color: Colors.grey,
                          fontSize: 14,
                          fontWeight: FontWeight.w600)),
                  Text("06:30",
                      style: TextStyle(color: Colors.black, fontSize: 18)),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}