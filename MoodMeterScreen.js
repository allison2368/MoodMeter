import React, { useState } from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import DefinitionDisplay from './DefinitionDisplay';

export default function MoodMeterScreen({navigation}) {
  const [selectedMood, setSelectedMood] = useState(null);

  const moods = [
    { label: 'High Energy Pleasant', color: '#ffcc00', screen: 'HighEnergyPleasant'},
    { label: 'High Energy Unpleasant', color: '#ff3300' , screen: 'HighEnergyUnpleasant'},
    { label: 'Low Energy Pleasant', color: '#33cc33' , screen: 'LowEnergyPleasant'},
    { label: 'Low Energy Unpleasant', color: '#3399ff', screen: 'LowEnergyUnpleasant'},
  ];
  const nextButtons = {
    'High Energy Pleasant': 'View High Energy Pleasant Emotions',
    'High Energy Unpleasant': 'View High Energy Unpleasant Emotions',
    'Low Energy Pleasant': 'View Low Energy Pleasant Emotions',
    'Low Energy Unpleasant': 'View Low Energy Unpleasant Emotions',
  };

  const handleMoodSelect = (mood) => {
    setSelectedMood(mood);
  };
  const handleNext = () => {
    console.log('Proceeding with mood:', selectedMood);

    const selectedMoodData = moods.find((m) => m.label === selectedMood);

    if (selectedMoodData) {
      console.log(`Navigating to: ${selectedMoodData.screen}`);
      navigation.navigate(selectedMoodData.screen);  // Ensure this matches App.js
    } else {
      console.log("No matching screen found for mood.");
    }
};



  return (
    <View style={styles.container}>
      <Text style={styles.header}>What do you feel right now?</Text>
      <View style={styles.grid}>
        {moods.map((mood, index) => (//iterates over each index, or mood
          <TouchableOpacity//creates a button for each of the bigger moods
            key={index}
            style={[styles.cell, { backgroundColor: mood.color }]}
            onPress={() => handleMoodSelect(mood.label)}
          >
            <Text style={styles.cellText}>{mood.label}</Text>
          </TouchableOpacity>
        ))}
      </View>
      {selectedMood && <DefinitionDisplay mood={selectedMood} />}
	  {selectedMood && (
        <TouchableOpacity style={styles.nextButton} onPress={handleNext}>
          <Text style={styles.nextButtonText}>{nextButtons[selectedMood]}</Text>
        </TouchableOpacity>
      )}
    </View>
  );

}
const styles = StyleSheet.create({
  container: { flex: 1, padding: 20 },
  header: { fontSize: 20, fontWeight: 'bold', marginBottom: 20 },
  grid: { flexDirection: 'row', flexWrap: 'wrap', justifyContent: 'center' },
  cell: {
    width: '45%',
    height: 100,
    margin: 5,
    justifyContent: 'center',
    alignItems: 'center',
    borderRadius: 10,
  },
  cellText: { color: '#fff', fontWeight: 'bold', textAlign: 'center' },
  nextButton: {
    marginTop: 20,
    paddingVertical: 15,
    paddingHorizontal: 40,
    backgroundColor: '#007bff',
    borderRadius: 10,
  },
  nextButtonText: { color: '#fff', fontWeight: 'bold', fontSize: 16 },
});
