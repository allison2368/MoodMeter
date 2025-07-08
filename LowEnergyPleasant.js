import React from 'react';
import { View, Text, StyleSheet } from 'react-native';

export default function LowEnergyPleasant() {
  return (
	<View style={styles.container}>
	  <Text style={styles.text}>You are feeling low energy pleasant! 😊</Text>
	</View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  text: { fontSize: 24, fontWeight: 'bold' },
});
