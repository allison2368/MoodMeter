import React from 'react';
import { View, Text, StyleSheet } from 'react-native';

const moodDescriptions = {
	'High Energy Pleasant': "Excitement, joy and elation",
	'High Energy Unpleasant': "Anger, frustration and anxiety",
	'Low Energy Pleasant': "Tranquility, satisfaction and calm",
	'Low Energy Unpleasant': "Boredom, sadness and despair"
  };
export default function DefinitionDisplay({ mood }) {
  return (
    <View style={styles.container}>
      <Text style={styles.title}>Definition</Text>
      <Text>{mood}: {moodDescriptions[mood] || "No description available" }</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { marginTop: 20, padding: 10, borderWidth: 1, borderColor: '#ccc', borderRadius: 5 },
  title: { fontSize: 16, fontWeight: 'bold', marginBottom: 5 },
});
