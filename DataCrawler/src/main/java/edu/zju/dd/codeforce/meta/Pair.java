package edu.zju.dd.codeforce.meta;

import java.util.Comparator;


public class Pair<T, E> {
	T first;
	E second;
	
	public Pair(){
		
	}
	
	public Pair(T first, E second){
		this.first = first;
		this.second = second;
	}
	
	public T getFirst() {
		return first;
	}
	public void setFirst(T first) {
		this.first = first;
	}
	public E getSecond() {
		return second;
	}
	public void setSecond(E second) {
		this.second = second;
	}
	
	/**
	 * order by second, ascending
	 */
	public static  Comparator<Pair<String, Double>> comparatorASC = 
		new Comparator<Pair<String, Double>>() {
		public int compare(Pair<String, Double> o1,
				Pair<String, Double> o2) {
			if (o1.second < o2.second)
				return -1;
			if (o1.second > o2.second)
				return 1;
			return 0;
		}
	};
	/**
	 * order by second, descending
	 */
	public static  Comparator<Pair<String, Double>> comparatorDESC = 
		new Comparator<Pair<String, Double>>() {
		public int compare(Pair<String, Double> o1,
				Pair<String, Double> o2) {
			if (o1.second > o2.second)
				return -1;
			if (o1.second < o2.second)
				return 1;
			return 0;
		}
	};
}

