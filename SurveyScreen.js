import React from 'react';
import {SelectList } from 'react-native-dropdown-select-list';
import { View, Text, TextInput, Button, StyleSheet } from 'react-native';

export default function SurveyScreen({ navigation }) {
  const [selected, setSelected] = React.useState("");
  const data_age = [
    { key: '1', value: '5-9' },
	{ key: '2', value: '10-15' },
    { key: '3', value: '16-20' },
	{ key: '4', value: '21-26' },
  ];
  const data_student = [
	{key: '1', value: 'Yes'},
	{key: '2', value: 'No'}
  ];

  const handleSkip = () => {
    navigation.navigate('MoodMeter');
  };

  return (
    <View style={styles.container}>
      <Text style={styles.header}>Survey Questions</Text>
	  <Text style={styles.label}>Are you currently a student?</Text>
	  <SelectList 
        data={data_student} 
        setSelected={setSelected} 
        placeholder="Select Yes or No"
        boxStyles={styles.dropdown}
      />
      <Text style={styles.label}>Age Range:</Text>
      <SelectList 
        data={data_age} 
        setSelected={setSelected} 
        placeholder="Select your age range"
        boxStyles={styles.dropdown}
      />
      <Button title="Next" onPress={handleSkip} />
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, padding: 20, justifyContent: 'center', backgroundColor: '#fff' },
  header: { fontSize: 20, fontWeight: 'bold', marginBottom: 20, textAlign: 'center' },
  label: { fontSize: 16, marginBottom: 10 },
  input: { 
    borderWidth: 1, 
    borderColor: '#ccc', 
    padding: 10, 
    marginBottom: 20, 
    borderRadius: 5 
  },
  dropdown: {
    borderWidth: 1,
    borderColor: '#ccc',
    marginBottom: 20,
    borderRadius: 5,
    padding: 10,
  },
});
